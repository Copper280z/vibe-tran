// src/io/inp_parser.cpp
// CalculiX/Abaqus .inp file parser implementation.

#include "io/inp_parser.hpp"
#include "core/exceptions.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <format>
#include <sstream>
#include <unordered_map>

namespace vibestran {

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return s;
}

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(trim(field));
    }
    return fields;
}

// ── Keyword block representation ────────────────────────────────────────────

struct KeywordBlock {
    std::string keyword;   // uppercase, e.g. "NODE", "SOLID SECTION"
    std::unordered_map<std::string, std::string> params; // KEY=VALUE pairs (keys uppercase)
    std::vector<std::string> data_lines;
    int line_num{0};       // line number of the keyword line
};

// ── ParseContext ─────────────────────────────────────────────────────────────

struct InpParser::ParseContext {
    Model model;
    int line_num = 0;

    // Named sets
    std::unordered_map<std::string, std::vector<NodeId>> nsets;
    std::unordered_map<std::string, std::vector<ElementId>> elsets;

    // Material accumulation
    std::unordered_map<std::string, Mat1> materials_by_name;
    std::string current_material_name;
    int next_material_id = 1;
    int next_property_id = 1;

    // Track which material names have been finalized (assigned MaterialId)
    std::unordered_map<std::string, MaterialId> finalized_materials;

    // Element lookup for section assignment
    std::unordered_map<int, size_t> element_id_to_index;

    // Step tracking
    int current_step = 0;
    bool in_step = false;
    SubCase current_subcase;

    // Track whether model-level loads/SPCs exist
    bool has_model_level_loads = false;
    bool has_model_level_spcs = false;
};

// ── Forward declarations of keyword processors ──────────────────────────────

static void process_nodes(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_elements(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_nset(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_elset(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_material(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_elastic(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_density(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_expansion(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_solid_section(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_shell_section(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_boundary(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_cload(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_temperature(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_initial_conditions(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_step(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_static(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_node_output(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_element_output(InpParser::ParseContext& ctx, const KeywordBlock& block);
static void process_end_step(InpParser::ParseContext& ctx, const KeywordBlock& block);

// ── Parse a keyword line into keyword name + params ─────────────────────────

static KeywordBlock parse_keyword_line(const std::string& line, int line_num) {
    KeywordBlock block;
    block.line_num = line_num;

    // Remove leading '*'
    std::string content = line.substr(1);
    auto fields = split_csv(content);
    if (fields.empty()) return block;

    block.keyword = to_upper(trim(fields[0]));

    for (size_t i = 1; i < fields.size(); ++i) {
        std::string f = trim(fields[i]);
        auto eq = f.find('=');
        if (eq != std::string::npos) {
            std::string key = to_upper(trim(f.substr(0, eq)));
            std::string val = trim(f.substr(eq + 1));
            block.params[key] = val;
        } else {
            // Flag parameter (e.g., GENERATE)
            block.params[to_upper(f)] = "";
        }
    }

    return block;
}

// ── Resolve a node reference (could be numeric ID or set name) ──────────────

static std::vector<NodeId> resolve_node_ref(const InpParser::ParseContext& ctx,
                                            const std::string& ref) {
    // Try numeric first
    try {
        int id = std::stoi(ref);
        return {NodeId{id}};
    } catch (...) {}

    // Look up as set name (case-insensitive)
    std::string upper = to_upper(ref);
    auto it = ctx.nsets.find(upper);
    if (it != ctx.nsets.end()) return it->second;

    // Try original case
    it = ctx.nsets.find(ref);
    if (it != ctx.nsets.end()) return it->second;

    throw ParseError(std::format("Unknown node set '{}' referenced", ref));
}

// ── Resolve an element set reference ────────────────────────────────────────

static std::vector<ElementId> resolve_elset_ref(const InpParser::ParseContext& ctx,
                                                const std::string& ref) {
    std::string upper = to_upper(ref);
    auto it = ctx.elsets.find(upper);
    if (it != ctx.elsets.end()) return it->second;

    it = ctx.elsets.find(ref);
    if (it != ctx.elsets.end()) return it->second;

    throw ParseError(std::format("Unknown element set '{}' referenced", ref));
}

// ── Finalize material by name (assign ID, store in model) ───────────────────

static MaterialId finalize_material(InpParser::ParseContext& ctx,
                                    const std::string& name) {
    std::string upper = to_upper(name);

    // Already finalized?
    auto it = ctx.finalized_materials.find(upper);
    if (it != ctx.finalized_materials.end()) return it->second;

    // Find accumulated material data
    auto mat_it = ctx.materials_by_name.find(upper);
    if (mat_it == ctx.materials_by_name.end()) {
        throw ParseError(std::format("Material '{}' not defined", name));
    }

    MaterialId mid{ctx.next_material_id++};
    Mat1 mat = mat_it->second;
    mat.id = mid;

    // Derive G if not set
    if (mat.G == 0.0 && mat.E != 0.0 && mat.nu != 0.0) {
        mat.G = mat.E / (2.0 * (1.0 + mat.nu));
    }

    ctx.model.materials.insert_or_assign(mid, mat);
    ctx.finalized_materials.insert_or_assign(upper, mid);
    return mid;
}

// ── Element type mapping ────────────────────────────────────────────────────

struct ElemTypeInfo {
    ElementType type;
    int num_nodes;
};

static std::optional<ElemTypeInfo> map_element_type(const std::string& ccx_type) {
    std::string upper = to_upper(ccx_type);

    // Solid elements
    if (upper == "C3D8" || upper == "C3D8R")
        return ElemTypeInfo{ElementType::CHEXA8, 8};
    if (upper == "C3D4")
        return ElemTypeInfo{ElementType::CTETRA4, 4};
    if (upper == "C3D10")
        return ElemTypeInfo{ElementType::CTETRA10, 10};
    if (upper == "C3D6")
        return ElemTypeInfo{ElementType::CPENTA6, 6};

    // Shell elements
    if (upper == "S4" || upper == "S4R")
        return ElemTypeInfo{ElementType::CQUAD4, 4};
    if (upper == "S3" || upper == "S3R")
        return ElemTypeInfo{ElementType::CTRIA3, 3};

    // Unsupported but recognized
    if (upper == "C3D20" || upper == "C3D20R") {
        throw ParseError("C3D20/C3D20R (CHEXA20) elements are not supported in "
                        "the INP parser due to midside node reordering differences "
                        "between Abaqus and Nastran conventions");
    }

    return std::nullopt;
}

// ── Keyword processors ──────────────────────────────────────────────────────

static void process_nodes(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    // Check for NSET parameter (auto-add nodes to a set)
    std::string auto_nset;
    auto nset_it = block.params.find("NSET");
    if (nset_it != block.params.end()) {
        auto_nset = to_upper(nset_it->second);
    }

    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        if (fields.size() < 4) continue;

        int id = std::stoi(fields[0]);
        double x = std::stod(fields[1]);
        double y = std::stod(fields[2]);
        double z = std::stod(fields[3]);

        GridPoint gp;
        gp.id = NodeId{id};
        gp.cp = CoordId{0};
        gp.position = Vec3{x, y, z};
        gp.cd = CoordId{0};
        ctx.model.nodes[gp.id] = gp;

        if (!auto_nset.empty()) {
            ctx.nsets[auto_nset].push_back(NodeId{id});
        }
    }
}

static void process_elements(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    auto type_it = block.params.find("TYPE");
    if (type_it == block.params.end()) {
        throw ParseError(std::format("*ELEMENT at line {} missing TYPE parameter",
                                     block.line_num));
    }

    auto type_info = map_element_type(type_it->second);
    if (!type_info) {
        // Silently skip unrecognized element types
        return;
    }

    // Check for ELSET parameter
    std::string auto_elset;
    auto elset_it = block.params.find("ELSET");
    if (elset_it != block.params.end()) {
        auto_elset = to_upper(elset_it->second);
    }

    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        if (fields.empty()) continue;

        int eid = std::stoi(fields[0]);

        ElementData elem;
        elem.id = ElementId{eid};
        elem.type = type_info->type;
        elem.pid = PropertyId{0}; // assigned later by *SECTION

        for (size_t i = 1; i < fields.size(); ++i) {
            if (fields[i].empty()) continue;
            elem.nodes.push_back(NodeId{std::stoi(fields[i])});
        }

        if (static_cast<int>(elem.nodes.size()) != type_info->num_nodes) {
            throw ParseError(std::format(
                "*ELEMENT at line {}: element {} has {} nodes, expected {} for type {}",
                block.line_num, eid, elem.nodes.size(), type_info->num_nodes,
                type_it->second));
        }

        ctx.element_id_to_index[eid] = ctx.model.elements.size();
        ctx.model.elements.push_back(std::move(elem));

        if (!auto_elset.empty()) {
            ctx.elsets[auto_elset].push_back(ElementId{eid});
        }
    }
}

static void parse_set_data(const KeywordBlock& block,
                           bool generate,
                           std::vector<int>& ids) {
    if (generate) {
        // GENERATE: each line is start, end, increment
        for (const auto& line : block.data_lines) {
            auto fields = split_csv(line);
            if (fields.size() < 2) continue;
            int start = std::stoi(fields[0]);
            int end = std::stoi(fields[1]);
            int inc = (fields.size() >= 3 && !fields[2].empty()) ? std::stoi(fields[2]) : 1;
            for (int i = start; i <= end; i += inc) {
                ids.push_back(i);
            }
        }
    } else {
        // Explicit list: comma-separated IDs across data lines
        for (const auto& line : block.data_lines) {
            auto fields = split_csv(line);
            for (const auto& f : fields) {
                if (f.empty()) continue;
                ids.push_back(std::stoi(f));
            }
        }
    }
}

static void process_nset(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    auto name_it = block.params.find("NSET");
    if (name_it == block.params.end()) {
        throw ParseError(std::format("*NSET at line {} missing NSET parameter",
                                     block.line_num));
    }

    std::string name = to_upper(name_it->second);
    bool generate = block.params.count("GENERATE") > 0;

    std::vector<int> ids;
    parse_set_data(block, generate, ids);

    for (int id : ids) {
        ctx.nsets[name].push_back(NodeId{id});
    }
}

static void process_elset(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    auto name_it = block.params.find("ELSET");
    if (name_it == block.params.end()) {
        throw ParseError(std::format("*ELSET at line {} missing ELSET parameter",
                                     block.line_num));
    }

    std::string name = to_upper(name_it->second);
    bool generate = block.params.count("GENERATE") > 0;

    std::vector<int> ids;
    parse_set_data(block, generate, ids);

    for (int id : ids) {
        ctx.elsets[name].push_back(ElementId{id});
    }
}

static void process_material(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    auto name_it = block.params.find("NAME");
    if (name_it == block.params.end()) {
        throw ParseError(std::format("*MATERIAL at line {} missing NAME parameter",
                                     block.line_num));
    }

    ctx.current_material_name = to_upper(name_it->second);
    // Initialize empty material
    ctx.materials_by_name[ctx.current_material_name] = Mat1{};
}

static void process_elastic(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    if (ctx.current_material_name.empty()) {
        throw ParseError(std::format("*ELASTIC at line {} outside *MATERIAL block",
                                     block.line_num));
    }

    if (block.data_lines.empty()) {
        throw ParseError(std::format("*ELASTIC at line {} has no data", block.line_num));
    }

    auto fields = split_csv(block.data_lines[0]);
    if (fields.size() < 2) {
        throw ParseError(std::format("*ELASTIC at line {} needs E and nu", block.line_num));
    }

    auto& mat = ctx.materials_by_name[ctx.current_material_name];
    mat.E = std::stod(fields[0]);
    mat.nu = std::stod(fields[1]);
}

static void process_density(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    if (ctx.current_material_name.empty()) {
        throw ParseError(std::format("*DENSITY at line {} outside *MATERIAL block",
                                     block.line_num));
    }

    if (block.data_lines.empty()) return;

    auto fields = split_csv(block.data_lines[0]);
    if (fields.empty()) return;

    ctx.materials_by_name[ctx.current_material_name].rho = std::stod(fields[0]);
}

static void process_expansion(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    if (ctx.current_material_name.empty()) {
        throw ParseError(std::format("*EXPANSION at line {} outside *MATERIAL block",
                                     block.line_num));
    }

    if (block.data_lines.empty()) return;

    auto fields = split_csv(block.data_lines[0]);
    if (fields.empty()) return;

    ctx.materials_by_name[ctx.current_material_name].A = std::stod(fields[0]);
}

static void process_solid_section(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    auto elset_it = block.params.find("ELSET");
    auto mat_it = block.params.find("MATERIAL");

    if (elset_it == block.params.end()) {
        throw ParseError(std::format("*SOLID SECTION at line {} missing ELSET", block.line_num));
    }
    if (mat_it == block.params.end()) {
        throw ParseError(std::format("*SOLID SECTION at line {} missing MATERIAL", block.line_num));
    }

    MaterialId mid = finalize_material(ctx, mat_it->second);
    PropertyId pid{ctx.next_property_id++};

    PSolid ps;
    ps.pid = pid;
    ps.mid = mid;
    ps.cordm = 0;
    ps.isop = SolidFormulation::EAS;
    ctx.model.properties[pid] = ps;

    // Assign pid to all elements in the elset
    auto elems = resolve_elset_ref(ctx, elset_it->second);
    for (const auto& eid : elems) {
        auto idx_it = ctx.element_id_to_index.find(eid.value);
        if (idx_it != ctx.element_id_to_index.end()) {
            ctx.model.elements[idx_it->second].pid = pid;
        }
    }
}

static void process_shell_section(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    auto elset_it = block.params.find("ELSET");
    auto mat_it = block.params.find("MATERIAL");

    if (elset_it == block.params.end()) {
        throw ParseError(std::format("*SHELL SECTION at line {} missing ELSET", block.line_num));
    }
    if (mat_it == block.params.end()) {
        throw ParseError(std::format("*SHELL SECTION at line {} missing MATERIAL", block.line_num));
    }

    // First data line contains the thickness
    double thickness = 0.0;
    if (!block.data_lines.empty()) {
        auto fields = split_csv(block.data_lines[0]);
        if (!fields.empty() && !fields[0].empty()) {
            thickness = std::stod(fields[0]);
        }
    }

    MaterialId mid = finalize_material(ctx, mat_it->second);
    PropertyId pid{ctx.next_property_id++};

    PShell ps;
    ps.pid = pid;
    ps.mid1 = mid;
    ps.t = thickness;
    ps.mid2 = MaterialId{0};  // same as mid1
    ps.mid3 = MaterialId{0};
    ps.mid4 = MaterialId{0};
    ctx.model.properties[pid] = ps;

    // Assign pid to all elements in the elset
    auto elems = resolve_elset_ref(ctx, elset_it->second);
    for (const auto& eid : elems) {
        auto idx_it = ctx.element_id_to_index.find(eid.value);
        if (idx_it != ctx.element_id_to_index.end()) {
            ctx.model.elements[idx_it->second].pid = pid;
        }
    }
}

static void process_boundary(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    SpcSetId sid = ctx.in_step ? SpcSetId{ctx.current_step} : SpcSetId{1};

    if (!ctx.in_step) {
        ctx.has_model_level_spcs = true;
    }

    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        if (fields.size() < 2) continue;

        // First field: node ID or set name
        auto node_ids = resolve_node_ref(ctx, fields[0]);

        int first_dof = std::stoi(fields[1]);
        int last_dof = (fields.size() >= 3 && !fields[2].empty())
                       ? std::stoi(fields[2]) : first_dof;
        double value = (fields.size() >= 4 && !fields[3].empty())
                       ? std::stod(fields[3]) : 0.0;

        for (const auto& nid : node_ids) {
            DofSet dofs;
            for (int d = first_dof; d <= last_dof; ++d) {
                if (d >= 1 && d <= 6) {
                    dofs.mask |= static_cast<uint8_t>(1 << (d - 1));
                }
            }
            Spc spc;
            spc.sid = sid;
            spc.node = nid;
            spc.dofs = dofs;
            spc.value = value;
            ctx.model.spcs.push_back(spc);
        }
    }
}

static void process_cload(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    LoadSetId sid = ctx.in_step ? LoadSetId{ctx.current_step} : LoadSetId{1};

    if (!ctx.in_step) {
        ctx.has_model_level_loads = true;
    }

    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        if (fields.size() < 3) continue;

        auto node_ids = resolve_node_ref(ctx, fields[0]);
        int dof = std::stoi(fields[1]);
        double magnitude = std::stod(fields[2]);

        for (const auto& nid : node_ids) {
            ForceLoad f;
            f.sid = sid;
            f.node = nid;
            f.cid = CoordId{0};
            f.scale = 1.0;

            Vec3 dir{0, 0, 0};
            switch (dof) {
                case 1: dir.x = magnitude; break;
                case 2: dir.y = magnitude; break;
                case 3: dir.z = magnitude; break;
                default:
                    throw ParseError(std::format(
                        "*CLOAD: unsupported DOF {} (only 1-3 supported for forces)",
                        dof));
            }
            f.direction = dir;
            ctx.model.loads.push_back(f);
        }
    }
}

static void process_temperature(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    LoadSetId sid = ctx.in_step ? LoadSetId{ctx.current_step} : LoadSetId{1};

    if (!ctx.in_step) {
        ctx.has_model_level_loads = true;
    }

    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        if (fields.size() < 2) continue;

        auto node_ids = resolve_node_ref(ctx, fields[0]);
        double temp = std::stod(fields[1]);

        for (const auto& nid : node_ids) {
            TempLoad tl;
            tl.sid = sid;
            tl.node = nid;
            tl.temperature = temp;
            ctx.model.loads.push_back(tl);
        }
    }
}

static void process_initial_conditions(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    // *INITIAL CONDITIONS, TYPE=TEMPERATURE
    auto type_it = block.params.find("TYPE");
    if (type_it == block.params.end()) return;
    if (to_upper(type_it->second) != "TEMPERATURE") return;

    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        if (fields.size() < 2) continue;

        double temp = std::stod(fields[1]);
        // Store as default temperature for set 1
        ctx.model.tempd[1] = temp;
    }
}

static void process_step(InpParser::ParseContext& ctx, const KeywordBlock& /*block*/) {
    ctx.current_step++;
    ctx.in_step = true;
    ctx.current_subcase = SubCase{};
    ctx.current_subcase.id = ctx.current_step;
}

static void process_static(InpParser::ParseContext& ctx, const KeywordBlock& /*block*/) {
    ctx.model.analysis.sol = SolutionType::LinearStatic;
}

static void process_node_output(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    if (!ctx.in_step) return;

    // Check data lines for U (displacements)
    bool has_u = block.data_lines.empty(); // empty means all outputs
    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        for (const auto& f : fields) {
            if (to_upper(f) == "U") has_u = true;
        }
    }

    if (has_u) {
        if (block.keyword == "NODE FILE") {
            ctx.current_subcase.disp_plot = true;
        } else { // NODE PRINT
            ctx.current_subcase.disp_print = true;
        }
    }
}

static void process_element_output(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    if (!ctx.in_step) return;

    // Check data lines for S (stresses)
    bool has_s = block.data_lines.empty();
    for (const auto& line : block.data_lines) {
        auto fields = split_csv(line);
        for (const auto& f : fields) {
            if (to_upper(f) == "S") has_s = true;
        }
    }

    if (has_s) {
        if (block.keyword == "EL FILE") {
            ctx.current_subcase.stress_plot = true;
        } else { // EL PRINT
            ctx.current_subcase.stress_print = true;
        }
    }
}

static void process_end_step(InpParser::ParseContext& ctx, const KeywordBlock& /*block*/) {
    if (!ctx.in_step) return;

    ctx.current_subcase.load_set = LoadSetId{ctx.current_step};
    ctx.current_subcase.spc_set = SpcSetId{ctx.current_step};

    // If there are model-level SPCs but no step-level SPCs, reference set 1
    if (ctx.has_model_level_spcs) {
        SpcSetId step_sid{ctx.current_step};
        bool has_step_spcs = std::any_of(ctx.model.spcs.begin(), ctx.model.spcs.end(),
            [&](const Spc& spc) { return spc.sid == step_sid; });
        if (!has_step_spcs) {
            ctx.current_subcase.spc_set = SpcSetId{1};
        }
    }

    // Same for loads
    if (ctx.has_model_level_loads) {
        LoadSetId step_lid{ctx.current_step};
        bool has_step_loads = std::any_of(ctx.model.loads.begin(), ctx.model.loads.end(),
            [&](const Load& load) {
                return std::visit([&](const auto& l) { return l.sid == step_lid; }, load);
            });
        if (!has_step_loads) {
            ctx.current_subcase.load_set = LoadSetId{1};
        }
    }

    ctx.model.analysis.subcases.push_back(ctx.current_subcase);
    ctx.in_step = false;
}

// ── Main dispatch ───────────────────────────────────────────────────────────

static void dispatch_block(InpParser::ParseContext& ctx, const KeywordBlock& block) {
    const auto& kw = block.keyword;

    if (kw == "NODE")                  process_nodes(ctx, block);
    else if (kw == "ELEMENT")          process_elements(ctx, block);
    else if (kw == "NSET")             process_nset(ctx, block);
    else if (kw == "ELSET")            process_elset(ctx, block);
    else if (kw == "MATERIAL")         process_material(ctx, block);
    else if (kw == "ELASTIC")          process_elastic(ctx, block);
    else if (kw == "DENSITY")          process_density(ctx, block);
    else if (kw == "EXPANSION")        process_expansion(ctx, block);
    else if (kw == "SOLID SECTION")    process_solid_section(ctx, block);
    else if (kw == "SHELL SECTION")    process_shell_section(ctx, block);
    else if (kw == "BOUNDARY")         process_boundary(ctx, block);
    else if (kw == "CLOAD")            process_cload(ctx, block);
    else if (kw == "TEMPERATURE")      process_temperature(ctx, block);
    else if (kw == "INITIAL CONDITIONS") process_initial_conditions(ctx, block);
    else if (kw == "STEP")             process_step(ctx, block);
    else if (kw == "STATIC")           process_static(ctx, block);
    else if (kw == "NODE FILE" || kw == "NODE PRINT")
                                       process_node_output(ctx, block);
    else if (kw == "EL FILE" || kw == "EL PRINT")
                                       process_element_output(ctx, block);
    else if (kw == "END STEP")         process_end_step(ctx, block);
    // else: silently skip unrecognized keywords
}

// ── Core parsing logic ──────────────────────────────────────────────────────

static Model parse_inp(std::istream& in) {
    InpParser::ParseContext ctx;
    ctx.model.analysis.sol = SolutionType::LinearStatic;

    // Read all lines
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        // Strip \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }

    // Parse into keyword blocks and dispatch
    KeywordBlock current_block;
    bool in_block = false;

    auto flush_block = [&]() {
        if (in_block) {
            dispatch_block(ctx, current_block);
            in_block = false;
        }
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        const auto& l = lines[i];
        int lnum = static_cast<int>(i + 1);

        // Skip empty lines
        if (trim(l).empty()) continue;

        // Skip comment lines
        if (l.size() >= 2 && l[0] == '*' && l[1] == '*') continue;

        // Keyword line
        if (!l.empty() && l[0] == '*') {
            flush_block();
            current_block = parse_keyword_line(l, lnum);
            in_block = true;
            continue;
        }

        // Data line
        if (in_block) {
            current_block.data_lines.push_back(l);
        }
    }

    flush_block();

    // If no *STEP was encountered, create a default subcase
    if (ctx.model.analysis.subcases.empty()) {
        SubCase sc;
        sc.id = 1;
        sc.load_set = LoadSetId{1};
        sc.spc_set = SpcSetId{1};
        sc.disp_print = true;
        sc.disp_plot = true;
        sc.stress_print = true;
        sc.stress_plot = true;
        ctx.model.analysis.subcases.push_back(sc);
    }

    // Apply TEMPD to subcases that reference it
    for (auto& sc : ctx.model.analysis.subcases) {
        if (sc.t_ref == 0.0) {
            auto it = ctx.model.tempd.find(sc.load_set.value);
            if (it != ctx.model.tempd.end()) {
                sc.t_ref = it->second;
            } else {
                // Try set 1 as fallback
                auto it1 = ctx.model.tempd.find(1);
                if (it1 != ctx.model.tempd.end()) {
                    sc.t_ref = it1->second;
                }
            }
        }
    }

    return std::move(ctx.model);
}

// ── Public API ──────────────────────────────────────────────────────────────

Model InpParser::parse_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw ParseError(std::format("Cannot open INP file: {}", path.string()));
    }
    return parse_inp(in);
}

Model InpParser::parse_string(const std::string& content) {
    std::istringstream in(content);
    return parse_inp(in);
}

Model InpParser::parse_stream(std::istream& in) {
    return parse_inp(in);
}

} // namespace vibestran
