// src/io/bdf_parser.cpp
// Nastran BDF parser supporting small-field (8-char), large-field (16-char),
// and free-field (comma-separated) formats. Handles continuation cards,
// case control section, and bulk data section.

#include "io/bdf_parser.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <format>
#include <fstream>
#include <sstream>

namespace nastran {

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
      } else if (kw.starts_with("SPC")) {
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
      } else if (kw.starts_with("TEMP(LOAD)") || kw.starts_with("TEMP(INIT)")) {
        // Not used in v1 but consume gracefully
      }
    }
    if (in_sc)
      ctx.model.analysis.subcases.push_back(cur_sc);
    if (ctx.model.analysis.subcases.empty()) {
      // Default single subcase
      ctx.model.analysis.subcases.push_back(
          SubCase{1, "DEFAULT", LoadSetId{1}, SpcSetId{1}});
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
    // Detect large-field (starts with * or +*)
    bool is_large = !l.empty() && l[0] == '*';

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
    // Uppercase, strip leading '*' (large-field prefix), trailing whitespace
    std::transform(kw.begin(), kw.end(), kw.begin(), ::toupper);
    if (!kw.empty() && kw[0] == '*')
      kw.erase(0, 1);
    while (!kw.empty() && std::isspace((unsigned char)kw.back()))
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
  ctx.model.properties[p.pid] = p;
}

void BdfParser::process_psolid(ParseContext &ctx,
                               const std::vector<std::string> &f) {
  // PSOLID, PID, MID, CORDM, IN, STRESS, ISOP, FCTN
  PSolid p;
  p.pid = PropertyId(parse_int(f[1], ctx.line_num));
  p.mid = MaterialId(parse_int(f[2], ctx.line_num));
  p.cordm = f[3].empty() ? 0 : parse_int(f[3], ctx.line_num);
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
  // CTETRA, EID, PID, G1, G2, G3, G4
  ElementData e;
  e.id = ElementId(parse_int(f[1], ctx.line_num));
  e.pid = PropertyId(parse_int(f[2], ctx.line_num));
  e.type = ElementType::CTETRA4;
  for (int i = 0; i < 4; ++i)
    e.nodes.push_back(NodeId(parse_int(f[3 + i], ctx.line_num)));
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

} // namespace nastran
