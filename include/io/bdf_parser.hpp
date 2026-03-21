#pragma once
// include/io/bdf_parser.hpp
// Nastran BDF (Bulk Data File) parser.
//
// Supports:
//   - Free-field format (comma-separated)
//   - Small-field format (8-char fixed columns)
//   - Large-field format (16-char, double-width with *)
//   - Continuation cards (+ or *)
//   - INCLUDE statements (recursive)
//   - Comment lines ($)
//
// Parsed card types: GRID, MAT1, PSHELL, PSOLID,
//                    CQUAD4, CTRIA3, CHEXA, CTETRA,
//                    FORCE, MOMENT, TEMP, TEMPD,
//                    SPC, SPC1,
//                    SOL, SUBCASE, LOAD, SPC (case control)

#include "core/model.hpp"
#include <filesystem>
#include <string>
#include <istream>

namespace vibetran {

class BdfParser {
public:
    /// Parse a BDF file and return the populated model.
    [[nodiscard]] static Model parse_file(const std::filesystem::path& path);

    /// Parse BDF content from a string (useful in tests)
    [[nodiscard]] static Model parse_string(const std::string& content);

    /// Parse from an input stream
    [[nodiscard]] static Model parse_stream(std::istream& in);

private:
    // Internal state during parsing
    struct ParseContext;

    static void parse_case_control(ParseContext& ctx);
    static void parse_bulk_data(ParseContext& ctx);

    // Card processors
    static void process_grid    (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_mat1    (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_pshell  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_psolid  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_cquad4  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_ctria3  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_chexa   (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_ctetra  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_cpenta  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_force   (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_moment  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_temp    (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_tempd   (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_spc     (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_spc1    (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_mpc     (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_mpcadd  (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_cord2x  (ParseContext& ctx, const std::vector<std::string>& fields, CoordType ctype);
    static void process_cord1x  (ParseContext& ctx, const std::vector<std::string>& fields, CoordType ctype);
    static void process_rbe2    (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_rbe3    (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_param   (ParseContext& ctx, const std::vector<std::string>& fields);
    static void process_eigrl   (ParseContext& ctx, const std::vector<std::string>& fields);

    // Field parsing helpers
    static double parse_double(const std::string& s, int line);
    static int    parse_int   (const std::string& s, int line);

    // Line splitting
    static std::vector<std::string> split_small_field(const std::string& line);
    static std::vector<std::string> split_large_field(const std::string& line,
                                                        const std::string& cont);
    static std::vector<std::string> split_free_field (const std::string& line);
};

} // namespace vibetran
