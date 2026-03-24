// src/io/bdf_parser.cpp
// Nastran BDF parser supporting small-field (8-char), large-field (16-char),
// and free-field (comma-separated) formats. Handles continuation cards,
// case control section, and bulk data section.

#include "io/bdf_parser.hpp"
#include "core/coord_sys.hpp"
#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <sstream>

namespace vibestran {

// ── ParseContext
// ──────────────────────────────────────────────────────────────

struct BdfParser::ParseContext {
  Model model;
  int line_num{0};

  // Collected TEMPD (default temperature) per set
  std::unordered_map<int, double> tempd_map;
};

// ── Public entry points
// ───────────────────────────────────────────────────────

Model BdfParser::parse_file(const std::filesystem::path &path) {
  std::ifstream f(path);
  if (!f)
    throw ParseError(std::format("Cannot open BDF file: {}", path.string()));
  return parse_stream(f);
}

Model BdfParser::parse_string(const std::string &content) {
  std::istringstream ss(content);
  return parse_stream(ss);
}

Model BdfParser::parse_stream(std::istream &in) {
  ParseContext ctx;

  // Read all lines, split into case control and bulk data sections
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    // Normalize line endings
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    lines.push_back(line);
  }

  // Find SOL, BEGIN BULK, ENDDATA markers
  int begin_bulk_line = -1;
  int enddata_line = static_cast<int>(lines.size());

  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    std::string upper = lines[i];
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    // Strip leading whitespace for keyword detection
    size_t first = upper.find_first_not_of(" \t");
    std::string_view keyword = (first == std::string::npos)
                                   ? ""
                                   : std::string_view(upper).substr(first);

    if (keyword.starts_with("BEGIN BULK") ||
        keyword.starts_with("BEGIN  BULK")) {
      begin_bulk_line = i + 1;
    } else if (keyword.starts_with("ENDDATA")) {
      enddata_line = i;
      break;
    } else if (begin_bulk_line == -1 && keyword.starts_with("SOL ")) {
      // Parse SOL in case control
      int sol_num = 0;
      std::string num_str = std::string(keyword.substr(4));
      // trim
      size_t ns = num_str.find_first_not_of(" \t");
      if (ns != std::string::npos)
        num_str = num_str.substr(ns);
      try {
        sol_num = std::stoi(num_str);
      } catch (...) {
      }
      if (sol_num == 101)
        ctx.model.analysis.sol = SolutionType::LinearStatic;
      else if (sol_num == 103)
        ctx.model.analysis.sol = SolutionType::Modal;
    }
  }

  if (begin_bulk_line == -1)
    begin_bulk_line = 0; // Assume entire file is bulk data

  // ── Parse case control (lines before BEGIN BULK) ─────────────────────────
  {
    // Simple tokenizer for case control
    SubCase cur_sc;
    bool in_sc = false;

    for (int i = 0; i < begin_bulk_line; ++i) {
      std::string upper = lines[i];
      std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
      // Strip comment
      size_t dlr = upper.find('$');
      if (dlr != std::string::npos)
        upper.resize(dlr);
      // Trim
      while (!upper.empty() && std::isspace((unsigned char)upper.back()))
        upper.pop_back();
      size_t first = upper.find_first_not_of(" \t");
      if (first == std::string::npos)
        continue;
      std::string kw = upper.substr(first);

      if (kw.starts_with("SUBCASE")) {
        if (in_sc)
          ctx.model.analysis.subcases.push_back(cur_sc);
        cur_sc = SubCase{};
        in_sc = true;
        // Parse ID
        size_t eq = kw.find('=');
        if (eq != std::string::npos) {
          try {
            cur_sc.id = std::stoi(kw.substr(eq + 1));
          } catch (...) {
          }
        } else {
          // "SUBCASE 1" form
          size_t sp = kw.find(' ');
          if (sp != std::string::npos) {
            try {
              cur_sc.id = std::stoi(kw.substr(sp + 1));
            } catch (...) {
            }
          }
        }
      } else if (kw.starts_with("LOAD")) {
        size_t eq = kw.find('=');
        if (eq != std::string::npos)
          try {
            cur_sc.load_set = LoadSetId(std::stoi(kw.substr(eq + 1)));
          } catch (...) {
          }
      } else if (kw.starts_with("MPC") && kw.find('=') != std::string::npos
                 && !kw.starts_with("MPCADD")) {
        size_t eq = kw.find('=');
        try {
          cur_sc.mpc_set = MpcSetId(std::stoi(kw.substr(eq + 1)));
        } catch (...) {}
      } else if (kw.starts_with("SPC") && !kw.starts_with("SPCADD")) {
        size_t eq = kw.find('=');
        if (eq != std::string::npos)
          try {
            cur_sc.spc_set = SpcSetId(std::stoi(kw.substr(eq + 1)));
          } catch (...) {
          }
      } else if (kw.starts_with("LABEL")) {
        size_t eq = kw.find('=');
        if (eq != std::string::npos)
          cur_sc.label = kw.substr(eq + 1);
      } else if (kw.starts_with("TEMP(LOAD)") || kw.starts_with("TEMPERATURE(LOAD)")) {
        size_t eq = kw.find('=');
        if (eq != std::string::npos)
          try { cur_sc.temp_load_set = std::stoi(kw.substr(eq + 1)); } catch (...) {}
      } else if (kw.starts_with("TEMP(INIT)") || kw.starts_with("TEMPERATURE(INIT)")) {
        // Not used yet but consume gracefully
      } else if (kw.starts_with("DISPLACEMENT")) {
        // Syntax: DISPLACEMENT[(PRINT[,PLOT])] = ALL|NONE|<set_id>
        // PRINT → F06/CSV output; PLOT → OP2 output.
        // No modifier is equivalent to PRINT (MSC Nastran default).
        // NONE clears both flags.
        size_t eq = kw.find('=');
        if (eq != std::string::npos) {
          std::string val = kw.substr(eq + 1);
          size_t ns = val.find_first_not_of(" \t");
          if (ns != std::string::npos) val = val.substr(ns);
          if (val.starts_with("NONE")) {
            cur_sc.disp_print = false;
            cur_sc.disp_plot  = false;
          } else {
            // Parse modifier list between '(' and ')'
            size_t lp = kw.find('(');
            size_t rp = kw.find(')');
            if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
              std::string mods = kw.substr(lp + 1, rp - lp - 1);
              if (mods.find("PRINT") != std::string::npos) cur_sc.disp_print = true;
              if (mods.find("PLOT")  != std::string::npos) cur_sc.disp_plot  = true;
            } else {
              // No modifier list → PRINT (text) output only
              cur_sc.disp_print = true;
            }
          }
        }
      } else if (kw.starts_with("METHOD")) {
        // METHOD = <sid>  — references an EIGRL card SID
        size_t eq = kw.find('=');
        if (eq != std::string::npos)
          try { cur_sc.eigrl_id = std::stoi(kw.substr(eq + 1)); } catch (...) {}
      } else if (kw.starts_with("EIGENVECTOR") || kw.starts_with("VECTOR")) {
        // EIGENVECTOR[(PRINT[,PLOT])] = ALL
        // VECTOR is an alias for EIGENVECTOR
        size_t eq = kw.find('=');
        if (eq != std::string::npos) {
          std::string val = kw.substr(eq + 1);
          size_t ns = val.find_first_not_of(" \t");
          if (ns != std::string::npos) val = val.substr(ns);
          if (!val.starts_with("NONE")) {
            size_t lp = kw.find('(');
            size_t rp = kw.find(')');
            if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
              std::string mods = kw.substr(lp + 1, rp - lp - 1);
              if (mods.find("PRINT") != std::string::npos) cur_sc.eigvec_print = true;
              if (mods.find("PLOT")  != std::string::npos) cur_sc.eigvec_plot  = true;
            } else {
              cur_sc.eigvec_print = true;
            }
          }
        }
      } else if (kw.starts_with("STRESS") || kw.starts_with("STRAIN")) {
        // Syntax: STRESS[(PRINT[,PLOT])] = ALL|NONE|<set_id>
        size_t eq = kw.find('=');
        if (eq != std::string::npos) {
          std::string val = kw.substr(eq + 1);
          size_t ns = val.find_first_not_of(" \t");
          if (ns != std::string::npos) val = val.substr(ns);
          if (val.starts_with("NONE")) {
            cur_sc.stress_print = false;
            cur_sc.stress_plot  = false;
          } else {
            size_t lp = kw.find('(');
            size_t rp = kw.find(')');
            if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
              std::string mods = kw.substr(lp + 1, rp - lp - 1);
              if (mods.find("PRINT") != std::string::npos) cur_sc.stress_print = true;
              if (mods.find("PLOT")  != std::string::npos) cur_sc.stress_plot  = true;
            } else {
              cur_sc.stress_print = true;
            }
          }
        }
      }
    }
    if (in_sc)
      ctx.model.analysis.subcases.push_back(cur_sc);
    if (ctx.model.analysis.subcases.empty()) {
      // No SUBCASE block: global case control entries were collected in cur_sc.
      // Promote them as subcase 1 so METHOD, SPC, LOAD, etc. take effect.
      if (cur_sc.id == 1 && cur_sc.label.empty())
        cur_sc.label = "DEFAULT";
      if (cur_sc.spc_set.value == 0 && cur_sc.eigrl_id == 0 &&
          cur_sc.load_set.value == 0) {
        // Truly empty — fall back to old default
        ctx.model.analysis.subcases.push_back(
            SubCase{1, "DEFAULT", LoadSetId{1}, SpcSetId{1}});
      } else {
        ctx.model.analysis.subcases.push_back(cur_sc);
      }
    }
  }

  // ── Parse bulk data ───────────────────────────────────────────────────────
  // Collect logical cards (handling continuations)
  struct LogicalCard {
    std::vector<std::string> fields; // up to 10 fields per line, multi-line
    int first_line{0};
  };

  std::vector<LogicalCard> cards;

  auto is_comment = [](const std::string &l) {
    size_t f = l.find_first_not_of(" \t");
    return f == std::string::npos || l[f] == '$';
  };

  auto is_continuation = [](const std::string &l) -> bool {
    if (l.empty())
      return false;
    char c0 = l[0];
    if (c0 == '+')
      return true;
    if (c0 == '*') {
      // A bare '*' (cols 0-7 contain only '*' and spaces) is a large-field
      // continuation marker. '*GRID', '*MAT1', etc. are new large-field cards.
      for (size_t i = 1; i < std::min(l.size(), size_t(8)); ++i)
        if (!std::isspace((unsigned char)l[i]))
          return false;
      return true;
    }
    // Blank-label continuation: first 8 columns are all whitespace but the
    // line has non-whitespace content beyond that (standard Nastran small-field
    // continuation without an explicit '+' marker).
    if (std::isspace((unsigned char)c0)) {
      // Check that cols 0-7 are all whitespace
      size_t kw_end = std::min(l.size(), size_t(8));
      bool kw_blank = true;
      for (size_t i = 0; i < kw_end; ++i) {
        if (!std::isspace((unsigned char)l[i])) {
          kw_blank = false;
          break;
        }
      }
      if (kw_blank) {
        // Must have non-whitespace content after the keyword field
        for (size_t i = kw_end; i < l.size(); ++i)
          if (!std::isspace((unsigned char)l[i]))
            return true;
      }
    }
    return false;
  };

  for (int i = begin_bulk_line; i < enddata_line; ++i) {
    const std::string &l = lines[i];
    if (is_comment(l))
      continue;
    if (l.empty())
      continue;

    // Detect free-field (has commas)
    bool is_free = l.find(',') != std::string::npos;
    // Detect large-field: starts with '*' (e.g. *GRID) OR keyword field ends
    // with '*' (e.g. GRID* — standard MSC Nastran large-field notation).
    bool is_large = !l.empty() && l[0] == '*';
    if (!is_large && !is_free) {
      size_t kend = std::min(l.size(), size_t(8));
      size_t last = l.find_last_not_of(" \t", kend - 1);
      if (last != std::string::npos && l[last] == '*')
        is_large = true;
    }

    if (is_continuation(l))
      continue; // handled below

    LogicalCard card;
    card.first_line = i + 1;

    if (is_free)
      card.fields = split_free_field(l);
    else if (is_large)
      card.fields = split_large_field(l, "");
    else
      card.fields = split_small_field(l);

    // For free-format lines, the trailing field may be a continuation marker
    // (e.g., "+", "+BC") — strip it so card processors don't see it as data.
    if (is_free && !card.fields.empty()) {
      const std::string &last = card.fields.back();
      if (!last.empty() && last[0] == '+')
        card.fields.pop_back();
    }

    // Accumulate continuations
    int j = i + 1;
    while (j < enddata_line) {
      const std::string &next = lines[j];
      if (is_comment(next)) {
        ++j;
        continue;
      }
      if (!is_continuation(next))
        break;

      std::vector<std::string> cont_fields;
      if (next.find(',') != std::string::npos)
        cont_fields = split_free_field(next);
      else if (!next.empty() && next[0] == '*')
        cont_fields = split_large_field(next, "");
      else
        cont_fields = split_small_field(next);

      // For free-format continuations, strip trailing continuation marker
      if (next.find(',') != std::string::npos && !cont_fields.empty()) {
        const std::string &clast = cont_fields.back();
        if (!clast.empty() && clast[0] == '+')
          cont_fields.pop_back();
      }

      // Skip the continuation marker (field 0) and append fields 1-8
      for (size_t k = 1; k < cont_fields.size() && k <= 8; ++k)
        card.fields.push_back(cont_fields[k]);

      ++j;
    }
    i = j - 1; // advance outer loop

    // Pad to at least 10 fields for safe access
    while (card.fields.size() < 10)
      card.fields.push_back("");
    cards.push_back(std::move(card));
  }

  // ── Dispatch cards ────────────────────────────────────────────────────────
  for (const auto &card : cards) {
    if (card.fields.empty())
      continue;
    std::string kw = card.fields[0];
    // Uppercase, strip leading/trailing '*' (large-field prefix/suffix), whitespace
    std::transform(kw.begin(), kw.end(), kw.begin(), ::toupper);
    if (!kw.empty() && kw[0] == '*')
      kw.erase(0, 1);
    while (!kw.empty() && (kw.back() == '*' || std::isspace((unsigned char)kw.back())))
      kw.pop_back();

    ctx.line_num = card.first_line;
    try {
      if (kw == "GRID")
        process_grid(ctx, card.fields);
      else if (kw == "MAT1")
        process_mat1(ctx, card.fields);
      else if (kw == "PSHELL")
        process_pshell(ctx, card.fields);
      else if (kw == "PSOLID")
        process_psolid(ctx, card.fields);
      else if (kw == "CQUAD4")
        process_cquad4(ctx, card.fields);
      else if (kw == "CTRIA3")
        process_ctria3(ctx, card.fields);
      else if (kw == "CHEXA")
        process_chexa(ctx, card.fields);
      else if (kw == "CTETRA")
        process_ctetra(ctx, card.fields);
      else if (kw == "CPENTA")
        process_cpenta(ctx, card.fields);
      else if (kw == "FORCE")
        process_force(ctx, card.fields);
      else if (kw == "MOMENT")
        process_moment(ctx, card.fields);
      else if (kw == "TEMP")
        process_temp(ctx, card.fields);
      else if (kw == "TEMPD")
        process_tempd(ctx, card.fields);
      else if (kw == "SPC")
        process_spc(ctx, card.fields);
      else if (kw == "SPC1")
        process_spc1(ctx, card.fields);
      else if (kw == "MPC")
        process_mpc(ctx, card.fields);
      else if (kw == "MPCADD")
        process_mpcadd(ctx, card.fields);
      else if (kw == "CORD2R")
        process_cord2x(ctx, card.fields, CoordType::Rectangular);
      else if (kw == "CORD2C")
        process_cord2x(ctx, card.fields, CoordType::Cylindrical);
      else if (kw == "CORD2S")
        process_cord2x(ctx, card.fields, CoordType::Spherical);
      else if (kw == "CORD1R")
        process_cord1x(ctx, card.fields, CoordType::Rectangular);
      else if (kw == "CORD1C")
        process_cord1x(ctx, card.fields, CoordType::Cylindrical);
      else if (kw == "RBE2")
        process_rbe2(ctx, card.fields);
      else if (kw == "RBE3")
        process_rbe3(ctx, card.fields);
      else if (kw == "PARAM")
        process_param(ctx, card.fields);
      else if (kw == "EIGRL")
        process_eigrl(ctx, card.fields);
      // Silently ignore unrecognized cards
    } catch (const ParseError &) {
      throw;
    } catch (const std::exception &e) {
      throw ParseError(std::format("Line {}: error processing {}: {}",
                                   ctx.line_num, kw, e.what()));
    }
  }

  // Apply TEMPD to subcases that don't have explicit t_ref
  for (auto &sc : ctx.model.analysis.subcases) {
    if (sc.t_ref == 0.0 && ctx.tempd_map.count(sc.load_set.value))
      sc.t_ref = ctx.tempd_map.at(sc.load_set.value);
  }

  // Post-parse: apply PARAM,SHELLFORM to all PShell properties
  {
    auto it = ctx.model.params.find("SHELLFORM");
    if (it != ctx.model.params.end()) {
      std::string val = it->second;
      std::transform(val.begin(), val.end(), val.begin(), ::toupper);
      ShellFormulation sf = (val == "MINDLIN") ? ShellFormulation::MINDLIN
                                                : ShellFormulation::MITC4;
      for (auto &[pid, prop] : ctx.model.properties) {
        if (std::holds_alternative<PShell>(prop))
          std::get<PShell>(prop).shell_form = sf;
      }
    }
  }

  // Post-parse: resolve coordinate systems and transform node positions to basic
  ctx.model.resolve_coordinates();

  return ctx.model;
}

// ── Field splitting
// ───────────────────────────────────────────────────────────

std::vector<std::string> BdfParser::split_small_field(const std::string &line) {
  // Each column is 8 chars: [0-7]=keyword, [8-15]=f1, [16-23]=f2, ...,
  // [72-79]=cont
  std::vector<std::string> fields;
  auto get_field = [&](int start, int width) -> std::string {
    if (start >= static_cast<int>(line.size()))
      return "";
    int end = std::min(start + width, static_cast<int>(line.size()));
    std::string f = line.substr(start, end - start);
    // Trim
    size_t s = f.find_first_not_of(" \t");
    size_t e = f.find_last_not_of(" \t");
    return (s == std::string::npos) ? "" : f.substr(s, e - s + 1);
  };

  fields.push_back(get_field(0, 8)); // keyword
  for (int i = 0; i < 8; ++i)
    fields.push_back(get_field(8 + i * 8, 8)); // fields 1-8
  // Cols 72-79 hold the continuation label; not a data field.
  return fields;
}

std::vector<std::string> BdfParser::split_large_field(const std::string &line,
                                                      const std::string &) {
  // Large-field: keyword(8), f1(16), f2(16), f3(16), f4(16), cont(8)
  std::vector<std::string> fields;
  auto get_field = [&](int start, int width) -> std::string {
    if (start >= static_cast<int>(line.size()))
      return "";
    int end = std::min(start + width, static_cast<int>(line.size()));
    std::string f = line.substr(start, end - start);
    size_t s = f.find_first_not_of(" \t");
    size_t e = f.find_last_not_of(" \t");
    return (s == std::string::npos) ? "" : f.substr(s, e - s + 1);
  };
  fields.push_back(get_field(0, 8));
  for (int i = 0; i < 4; ++i)
    fields.push_back(get_field(8 + i * 16, 16));
  // Cols 72-79 hold the continuation label; not a data field.
  return fields;
}

std::vector<std::string> BdfParser::split_free_field(const std::string &line) {
  // Free-field: comma-separated, strip $ comments.
  // The 10th field (index 9, after keyword + 8 data fields) is a continuation
  // label, not a data field — strip it so it does not appear in the data.
  std::string clean;
  size_t dlr = line.find('$');
  clean = (dlr == std::string::npos) ? line : line.substr(0, dlr);

  std::vector<std::string> fields;
  std::stringstream ss(clean);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    size_t s = tok.find_first_not_of(" \t");
    size_t e = tok.find_last_not_of(" \t");
    fields.push_back((s == std::string::npos) ? "" : tok.substr(s, e - s + 1));
  }

  // Strip continuation marker: if there are 10+ fields and the 10th (index 9)
  // starts with '+' or '*', it is a continuation label, not data.
  if (fields.size() >= 10) {
    const std::string &f9 = fields[9];
    if (!f9.empty() && (f9[0] == '+' || f9[0] == '*'))
      fields.erase(fields.begin() + 9);
  }

  return fields;
}

// ── Numeric field parsing
// ─────────────────────────────────────────────────────

double BdfParser::parse_double(const std::string &s, int line) {
  if (s.empty())
    return 0.0;

  // Nastran allows "1.5E+3", "1.5+3" (implicit E), "1.5D+3" (Fortran D)
  std::string clean = s;
  // Replace D exponent with E
  std::replace_if(clean.begin(), clean.end(),
                  [](char c) { return c == 'D' || c == 'd'; }, 'E');

  // Handle implicit exponent: "1.5+3" → "1.5E+3"
  // Look for +/- not preceded by E or at the start
  for (size_t i = 1; i < clean.size(); ++i) {
    if ((clean[i] == '+' || clean[i] == '-') &&
        std::toupper(clean[i - 1]) != 'E') {
      clean.insert(i, 1, 'E');
      break;
    }
  }

  try {
    return std::stod(clean);
  } catch (...) {
    throw ParseError(
        std::format("Line {}: cannot parse double: '{}'", line, s));
  }
}

int BdfParser::parse_int(const std::string &s, int line) {
  if (s.empty())
    return 0;
  try {
    return std::stoi(s);
  } catch (...) {
    throw ParseError(std::format("Line {}: cannot parse int: '{}'", line, s));
  }
}

namespace {

bool field_is_integer_like(const std::string &s) {
  if (s.empty())
    return false;
  size_t i = 0;
  if (s[i] == '+' || s[i] == '-')
    ++i;
  if (i >= s.size())
    return false;
  for (; i < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return false;
  return true;
}

} // namespace

// ── Card processors
// ───────────────────────────────────────────────────────────

void BdfParser::process_grid(ParseContext &ctx,
                             const std::vector<std::string> &f) {
  // GRID, ID, CP, X1, X2, X3, CD, PS, SEID
  GridPoint g;
  g.id = NodeId(parse_int(f[1], ctx.line_num));
  g.cp = CoordId(f[2].empty() ? 0 : parse_int(f[2], ctx.line_num));
  g.position =
      Vec3(parse_double(f[3], ctx.line_num), parse_double(f[4], ctx.line_num),
           parse_double(f[5], ctx.line_num));
  g.cd = CoordId(f[6].empty() ? 0 : parse_int(f[6], ctx.line_num));
  ctx.model.nodes[g.id] = g;
}

void BdfParser::process_mat1(ParseContext &ctx,
                             const std::vector<std::string> &f) {
  // MAT1, MID, E, G, NU, RHO, A, TREF, GE
  Mat1 m;
  m.id = MaterialId(parse_int(f[1], ctx.line_num));
  m.E = parse_double(f[2], ctx.line_num);
  m.G = f[3].empty() ? 0.0 : parse_double(f[3], ctx.line_num);
  m.nu = f[4].empty() ? 0.0 : parse_double(f[4], ctx.line_num);
  m.rho = f[5].empty() ? 0.0 : parse_double(f[5], ctx.line_num);
  m.A = f[6].empty() ? 0.0 : parse_double(f[6], ctx.line_num);
  m.ref_temp = f[7].empty() ? 0.0 : parse_double(f[7], ctx.line_num);

  // Derive G from E, nu if not given
  if (m.G == 0.0 && m.E > 0 && m.nu > 0)
    m.G = m.E / (2.0 * (1.0 + m.nu));
  ctx.model.materials[m.id] = m;
}

void BdfParser::process_pshell(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // PSHELL, PID, MID1, T, MID2, 12I/T**3, MID3, TST, NSM, Z1, Z2, MID4
  PShell p;
  p.pid = PropertyId(parse_int(f[1], ctx.line_num));
  p.mid1 = MaterialId(parse_int(f[2], ctx.line_num));
  p.t = parse_double(f[3], ctx.line_num);
  p.mid2 =
      f[4].empty() ? MaterialId{0} : MaterialId(parse_int(f[4], ctx.line_num));
  p.twelveI_t3 = f[5].empty() ? 1.0 : parse_double(f[5], ctx.line_num);
  p.mid3 =
      f[6].empty() ? MaterialId{0} : MaterialId(parse_int(f[6], ctx.line_num));
  p.tst = f[7].empty() ? 0.833333 : parse_double(f[7], ctx.line_num);
  p.nsm = f[8].empty() ? 0.0 : parse_double(f[8], ctx.line_num);
  p.z1 = (f.size() > 9 && !f[9].empty()) ? parse_double(f[9], ctx.line_num)
                                         : -0.5 * p.t;
  p.z2 = (f.size() > 10 && !f[10].empty()) ? parse_double(f[10], ctx.line_num)
                                           : 0.5 * p.t;
  p.mid4 = (f.size() > 11 && !f[11].empty())
               ? MaterialId(parse_int(f[11], ctx.line_num))
               : MaterialId{0};
  ctx.model.properties[p.pid] = p;
}

void BdfParser::process_psolid(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // PSOLID, PID, MID, CORDM, IN, STRESS, ISOP, FCTN
  PSolid p;
  p.pid = PropertyId(parse_int(f[1], ctx.line_num));
  p.mid = MaterialId(parse_int(f[2], ctx.line_num));
  p.cordm = f[3].empty() ? 0 : parse_int(f[3], ctx.line_num);
  // f[6] is ISOP: "EAS" (default) or "SRI" selects integration formulation
  if (f.size() > 6 && !f[6].empty()) {
    std::string isop = f[6];
    std::transform(isop.begin(), isop.end(), isop.begin(), ::toupper);
    if (isop == "SRI")
      p.isop = SolidFormulation::SRI;
    else if (isop == "EAS")
      p.isop = SolidFormulation::EAS;
  }
  ctx.model.properties[p.pid] = p;
}

void BdfParser::process_cquad4(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // CQUAD4, EID, PID, G1, G2, G3, G4, THETA/MCID, ZOFFS
  ElementData e;
  e.id = ElementId(parse_int(f[1], ctx.line_num));
  e.pid = PropertyId(parse_int(f[2], ctx.line_num));
  e.type = ElementType::CQUAD4;
  for (int i = 0; i < 4; ++i)
    e.nodes.push_back(NodeId(parse_int(f[3 + i], ctx.line_num)));
  if (f.size() > 7 && !f[7].empty()) {
    if (field_is_integer_like(f[7]))
      e.mcid = CoordId(parse_int(f[7], ctx.line_num));
    else
      e.theta = parse_double(f[7], ctx.line_num);
  }
  if (f.size() > 8 && !f[8].empty())
    e.zoffs = parse_double(f[8], ctx.line_num);
  ctx.model.elements.push_back(std::move(e));
}

void BdfParser::process_ctria3(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // CTRIA3, EID, PID, G1, G2, G3, THETA/MCID, ZOFFS
  ElementData e;
  e.id = ElementId(parse_int(f[1], ctx.line_num));
  e.pid = PropertyId(parse_int(f[2], ctx.line_num));
  e.type = ElementType::CTRIA3;
  for (int i = 0; i < 3; ++i)
    e.nodes.push_back(NodeId(parse_int(f[3 + i], ctx.line_num)));
  if (f.size() > 6 && !f[6].empty()) {
    if (field_is_integer_like(f[6]))
      e.mcid = CoordId(parse_int(f[6], ctx.line_num));
    else
      e.theta = parse_double(f[6], ctx.line_num);
  }
  if (f.size() > 7 && !f[7].empty())
    e.zoffs = parse_double(f[7], ctx.line_num);
  ctx.model.elements.push_back(std::move(e));
}

void BdfParser::process_chexa(ParseContext &ctx,
                              const std::vector<std::string> &f) {
  // CHEXA, EID, PID, G1..G8 or G1..G20. Node count determines variant:
  // 8 nodes → CHEXA8, 20 nodes → CHEXA20. Nodes may span continuation lines.
  // Empty slots (e.g. blank continuation-label fields) are skipped.
  ElementData e;
  e.id = ElementId(parse_int(f[1], ctx.line_num));
  e.pid = PropertyId(parse_int(f[2], ctx.line_num));
  for (int i = 3; i < static_cast<int>(f.size()); ++i)
    if (!f[i].empty())
      e.nodes.push_back(NodeId(parse_int(f[i], ctx.line_num)));

  if (e.nodes.size() == 8)
    e.type = ElementType::CHEXA8;
  else if (e.nodes.size() > 8)
    throw ParseError(std::format("Line {}: CHEXA has {} nodes; CHEXA20 is not yet supported",
                                 ctx.line_num, e.nodes.size()));
  else
    throw ParseError(std::format("Line {}: CHEXA has {} nodes; expected 8",
                                 ctx.line_num, e.nodes.size()));
  ctx.model.elements.push_back(std::move(e));
}

void BdfParser::process_ctetra(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // CTETRA, EID, PID, G1..G4 (4-node) or G1..G10 (10-node, may span continuations)
  ElementData e;
  e.id = ElementId(parse_int(f[1], ctx.line_num));
  e.pid = PropertyId(parse_int(f[2], ctx.line_num));
  for (int i = 3; i < static_cast<int>(f.size()); ++i)
    if (!f[i].empty())
      e.nodes.push_back(NodeId(parse_int(f[i], ctx.line_num)));

  if (e.nodes.size() == 4)
    e.type = ElementType::CTETRA4;
  else if (e.nodes.size() == 10)
    e.type = ElementType::CTETRA10;
  else
    throw ParseError(std::format("Line {}: CTETRA has {} nodes; expected 4 or 10",
                                 ctx.line_num, e.nodes.size()));
  ctx.model.elements.push_back(std::move(e));
}

void BdfParser::process_cpenta(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // CPENTA, EID, PID, G1..G6 (6-node) or G1..G15 (15-node, may span continuations)
  ElementData e;
  e.id = ElementId(parse_int(f[1], ctx.line_num));
  e.pid = PropertyId(parse_int(f[2], ctx.line_num));
  for (int i = 3; i < static_cast<int>(f.size()); ++i)
    if (!f[i].empty())
      e.nodes.push_back(NodeId(parse_int(f[i], ctx.line_num)));

  if (e.nodes.size() == 6)
    e.type = ElementType::CPENTA6;
  else
    throw ParseError(std::format("Line {}: CPENTA has {} nodes; expected 6",
                                 ctx.line_num, e.nodes.size()));
  ctx.model.elements.push_back(std::move(e));
}

void BdfParser::process_force(ParseContext &ctx,
                              const std::vector<std::string> &f) {
  // FORCE, SID, G, CID, F, N1, N2, N3
  ForceLoad l;
  l.sid = LoadSetId(parse_int(f[1], ctx.line_num));
  l.node = NodeId(parse_int(f[2], ctx.line_num));
  l.cid = CoordId(f[3].empty() ? 0 : parse_int(f[3], ctx.line_num));
  l.scale = parse_double(f[4], ctx.line_num);
  l.direction =
      Vec3(parse_double(f[5], ctx.line_num), parse_double(f[6], ctx.line_num),
           parse_double(f[7], ctx.line_num));
  ctx.model.loads.emplace_back(l);
}

void BdfParser::process_moment(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // MOMENT, SID, G, CID, M, N1, N2, N3
  MomentLoad l;
  l.sid = LoadSetId(parse_int(f[1], ctx.line_num));
  l.node = NodeId(parse_int(f[2], ctx.line_num));
  l.cid = CoordId(f[3].empty() ? 0 : parse_int(f[3], ctx.line_num));
  l.scale = parse_double(f[4], ctx.line_num);
  l.direction =
      Vec3(parse_double(f[5], ctx.line_num), parse_double(f[6], ctx.line_num),
           parse_double(f[7], ctx.line_num));
  ctx.model.loads.emplace_back(l);
}

void BdfParser::process_temp(ParseContext &ctx,
                             const std::vector<std::string> &f) {
  // TEMP, SID, G1, T1, G2, T2, G3, T3
  int sid = parse_int(f[1], ctx.line_num);
  for (int i = 0; i < 3; ++i) {
    int ni = 2 + 2 * i;
    int ti = 3 + 2 * i;
    if (ni >= static_cast<int>(f.size()) || f[ni].empty())
      break;
    TempLoad l;
    l.sid = LoadSetId(sid);
    l.node = NodeId(parse_int(f[ni], ctx.line_num));
    l.temperature = parse_double(f[ti], ctx.line_num);
    ctx.model.loads.emplace_back(l);
  }
}

void BdfParser::process_tempd(ParseContext &ctx,
                              const std::vector<std::string> &f) {
  // TEMPD, SID, T (default temperature for elements)
  int sid = parse_int(f[1], ctx.line_num);
  double T = parse_double(f[2], ctx.line_num);
  ctx.tempd_map[sid] = T;
  ctx.model.tempd[sid] = T;
}

void BdfParser::process_spc(ParseContext &ctx,
                            const std::vector<std::string> &f) {
  // SPC, SID, G1, C1, D1, G2, C2, D2
  int sid = parse_int(f[1], ctx.line_num);
  for (int i = 0; i < 2; ++i) {
    int ni = 2 + 3 * i;
    int ci = 3 + 3 * i;
    int di = 4 + 3 * i;
    if (ni >= static_cast<int>(f.size()) || f[ni].empty())
      break;
    Spc spc;
    spc.sid = SpcSetId(sid);
    spc.node = NodeId(parse_int(f[ni], ctx.line_num));
    spc.dofs = DofSet::from_int(parse_int(f[ci], ctx.line_num));
    spc.value = (di < static_cast<int>(f.size()) && !f[di].empty())
                    ? parse_double(f[di], ctx.line_num)
                    : 0.0;
    ctx.model.spcs.push_back(spc);
  }
}

void BdfParser::process_spc1(ParseContext &ctx,
                             const std::vector<std::string> &f) {
  // SPC1, SID, C, G1, G2, G3, ... (or G1 THRU Gn)
  int sid = parse_int(f[1], ctx.line_num);
  DofSet dofs = DofSet::from_int(parse_int(f[2], ctx.line_num));

  // Collect node IDs, handle THRU
  std::vector<int> node_ids;
  bool thru_mode = false;
  int thru_start = 0;
  for (size_t i = 3; i < f.size(); ++i) {
    if (f[i].empty())
      continue;
    std::string upper = f[i];
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "THRU") {
      thru_mode = true;
      if (!node_ids.empty()) node_ids.pop_back(); // start was prematurely added
    } else if (thru_mode) {
      int end = parse_int(f[i], ctx.line_num);
      for (int n = thru_start; n <= end; ++n)
        node_ids.push_back(n);
      thru_mode = false;
    } else {
      thru_start = parse_int(f[i], ctx.line_num);
      node_ids.push_back(thru_start);
    }
  }

  for (int nid : node_ids) {
    Spc spc;
    spc.sid = SpcSetId(sid);
    spc.node = NodeId(nid);
    spc.dofs = dofs;
    spc.value = 0.0;
    ctx.model.spcs.push_back(spc);
  }
}

void BdfParser::process_param(ParseContext &ctx,
                              const std::vector<std::string> &f) {
  // PARAM, NAME, VALUE
  if (f.size() < 3 || f[1].empty()) return;
  std::string name = f[1];
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  ctx.model.params[name] = f[2];
}

// ── Coordinate system processors ──────────────────────────────────────────────

void BdfParser::process_cord2x(ParseContext &ctx,
                                const std::vector<std::string> &f,
                                CoordType ctype) {
  // CORD2R/CORD2C/CORD2S, CID, RID, A1, A2, A3, B1, B2, B3, C1, C2, C3
  CoordSys cs;
  cs.id   = CoordId(parse_int(f[1], ctx.line_num));
  cs.rid  = CoordId(f[2].empty() ? 0 : parse_int(f[2], ctx.line_num));
  cs.type = ctype;
  // Defining points in RID frame
  cs.pt_a = Vec3(parse_double(f[3], ctx.line_num),
                 parse_double(f[4], ctx.line_num),
                 parse_double(f[5], ctx.line_num));
  cs.pt_b = Vec3(parse_double(f[6], ctx.line_num),
                 parse_double(f[7], ctx.line_num),
                 parse_double(f[8], ctx.line_num));
  cs.pt_c = Vec3(parse_double(f[9], ctx.line_num),
                 parse_double(f[10], ctx.line_num),
                 parse_double(f[11], ctx.line_num));
  ctx.model.coord_systems[cs.id] = cs;
}

void BdfParser::process_cord1x(ParseContext &ctx,
                                const std::vector<std::string> &f,
                                CoordType ctype) {
  // CORD1R/CORD1C, CID, G1, G2, G3
  // G1=origin node, G2=Z-axis node, G3=XZ-plane node (all in basic)
  CoordSys cs;
  cs.id   = CoordId(parse_int(f[1], ctx.line_num));
  cs.rid  = CoordId{0};
  cs.type = ctype;
  cs.is_cord1   = true;
  cs.def_node_a = parse_int(f[2], ctx.line_num);
  cs.def_node_b = parse_int(f[3], ctx.line_num);
  cs.def_node_c = parse_int(f[4], ctx.line_num);
  ctx.model.coord_systems[cs.id] = cs;
}

// ── MPC processors ────────────────────────────────────────────────────────────

void BdfParser::process_mpc(ParseContext &ctx,
                             const std::vector<std::string> &f) {
  // MPC, SID, G1, C1, A1, G2, C2, A2, ...  (continuation cards add more terms)
  // Each group of 3 fields (G, C, A) is one term.
  int sid = parse_int(f[1], ctx.line_num);
  Mpc mpc;
  mpc.sid = MpcSetId(sid);

  // Fields start at index 2; groups of 3
  for (size_t i = 2; i + 2 < f.size(); i += 3) {
    if (f[i].empty())
      break;
    MpcTerm term;
    term.node  = NodeId(parse_int(f[i],     ctx.line_num));
    term.dof   = parse_int(f[i + 1], ctx.line_num);
    term.coeff = parse_double(f[i + 2], ctx.line_num);
    mpc.terms.push_back(term);
  }

  if (!mpc.terms.empty())
    ctx.model.mpcs.push_back(std::move(mpc));
}

void BdfParser::process_mpcadd(ParseContext &ctx,
                                const std::vector<std::string> &f) {
  // MPCADD, SID, SID1, SID2, ...
  // Merges multiple MPC sets into one.  We expand by re-labelling all terms
  // from each source set to the target SID.
  int target_sid = parse_int(f[1], ctx.line_num);

  for (size_t i = 2; i < f.size(); ++i) {
    if (f[i].empty())
      continue;
    int src_sid = parse_int(f[i], ctx.line_num);
    if (src_sid == target_sid)
      continue; // avoid infinite loop

    // Collect MPCs with src_sid and re-label to target_sid
    std::vector<Mpc> new_mpcs;
    for (const auto &mpc : ctx.model.mpcs) {
      if (mpc.sid.value == src_sid) {
        Mpc copy = mpc;
        copy.sid = MpcSetId(target_sid);
        new_mpcs.push_back(std::move(copy));
      }
    }
    for (auto &m : new_mpcs)
      ctx.model.mpcs.push_back(std::move(m));
  }
}

// ── Rigid element processors ──────────────────────────────────────────────────

void BdfParser::process_rbe2(ParseContext &ctx,
                              const std::vector<std::string> &f) {
  // RBE2, EID, GN, CM, GM1, GM2, ..., ALPHA
  Rbe2 rbe;
  rbe.eid = ElementId(parse_int(f[1], ctx.line_num));
  rbe.gn  = NodeId(parse_int(f[2], ctx.line_num));
  rbe.cm  = DofSet::from_int(parse_int(f[3], ctx.line_num));

  for (size_t i = 4; i < f.size(); ++i) {
    if (f[i].empty())
      continue;
    // Last field may be ALPHA (a real number) — skip if looks like float
    // Simple check: if it contains a '.' or 'E' or 'e', it's ALPHA
    const std::string &fld = f[i];
    bool is_real = fld.find('.') != std::string::npos ||
                   fld.find('E') != std::string::npos ||
                   fld.find('e') != std::string::npos;
    if (is_real)
      break; // ALPHA field — done
    rbe.gm.push_back(NodeId(parse_int(fld, ctx.line_num)));
  }

  ctx.model.rbe2s.push_back(std::move(rbe));
}

void BdfParser::process_rbe3(ParseContext &ctx,
                              const std::vector<std::string> &f) {
  // RBE3, EID, blank, REFGRID, REFC, WT1, C1, G1,1, G1,2, ..., WT2, C2, ...
  // The structure is: EID, (blank), REFGRID, REFC, then repeating weight groups:
  //   each group: weight, component_set, node1, node2, ...
  // Groups are separated by the next real-number weight field.
  Rbe3 rbe;
  rbe.eid      = ElementId(parse_int(f[1], ctx.line_num));
  // f[2] is blank (SPEC/reserved)
  rbe.ref_node = NodeId(parse_int(f[3], ctx.line_num));
  rbe.refc     = DofSet::from_int(parse_int(f[4], ctx.line_num));

  // Parse weight groups starting at field 5
  // A weight group starts with a real weight, then an integer component set,
  // then one or more integer node IDs.
  size_t i = 5;
  while (i < f.size()) {
    if (f[i].empty()) { ++i; continue; }

    // Try to parse as real weight
    double w = 0.0;
    try {
      w = parse_double(f[i], ctx.line_num);
    } catch (...) {
      ++i;
      continue;
    }
    ++i;

    if (i >= f.size() || f[i].empty()) break;
    int comp_int = parse_int(f[i], ctx.line_num);
    ++i;

    Rbe3WeightGroup wg;
    wg.weight    = w;
    wg.component = DofSet::from_int(comp_int);

    // Collect node IDs until next real field or end
    while (i < f.size()) {
      if (f[i].empty()) { ++i; continue; }
      const std::string &fld = f[i];
      bool is_real = fld.find('.') != std::string::npos ||
                     fld.find('E') != std::string::npos ||
                     fld.find('e') != std::string::npos;
      if (is_real)
        break; // start of next weight group
      // Check if it's an integer
      try {
        int nid = parse_int(fld, ctx.line_num);
        wg.nodes.push_back(NodeId(nid));
      } catch (...) {
        break;
      }
      ++i;
    }

    if (!wg.nodes.empty())
      rbe.weight_groups.push_back(std::move(wg));
  }

  ctx.model.rbe3s.push_back(std::move(rbe));
}

// ── EIGRL processor ───────────────────────────────────────────────────────────

void BdfParser::process_eigrl(ParseContext& ctx,
                               const std::vector<std::string>& f) {
  // EIGRL, SID, V1, V2, ND, MSGLVL, MAXSET, SHFSCL, NORM
  // Field indices: 0=EIGRL, 1=SID, 2=V1, 3=V2, 4=ND, 5=MSGLVL, 6=MAXSET, 7=SHFSCL, 8=NORM
  EigRL e;
  e.sid = parse_int(f[1], ctx.line_num);
  if (f.size() > 2 && !f[2].empty())
    try { e.v1 = parse_double(f[2], ctx.line_num); } catch (...) {}
  if (f.size() > 3 && !f[3].empty())
    try { e.v2 = parse_double(f[3], ctx.line_num); } catch (...) {}
  if (f.size() > 4 && !f[4].empty())
    try { e.nd = parse_int(f[4], ctx.line_num); } catch (...) {}
  // Field 5 = MSGLVL, 6 = MAXSET, 7 = SHFSCL — skip
  if (f.size() > 8 && !f[8].empty()) {
    std::string norm = f[8];
    std::transform(norm.begin(), norm.end(), norm.begin(), ::toupper);
    if (norm.find("MAX") != std::string::npos)
      e.norm = EigRL::Norm::Max;
    else
      e.norm = EigRL::Norm::Mass;
  }
  ctx.model.eigrls[e.sid] = e;
}

} // namespace vibestran
