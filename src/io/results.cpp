// src/io/results.cpp
#include "io/results.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <numbers>
#include <format>
#include <algorithm>

namespace vibetran {

// ── Principal stress helpers ──────────────────────────────────────────────────

void compute_principal_2d(double sx, double sy, double sxy,
                          double &major, double &minor, double &angle_deg) {
    double avg  = (sx + sy) * 0.5;
    double diff = (sx - sy) * 0.5;
    double R    = std::sqrt(diff * diff + sxy * sxy);
    major     = avg + R;
    minor     = avg - R;
    // angle of major principal axis from x in degrees
    angle_deg = 0.5 * std::atan2(2.0 * sxy, sx - sy) * (180.0 / std::numbers::pi);
}

// Jacobi eigenvalue algorithm for 3×3 symmetric matrix.
// On entry:  a[3][3] symmetric, v = identity.
// On exit:   a[i][i] = eigenvalue i, v[i] = eigenvector i (row-wise).
static void jacobi3(double a[3][3], double v[3][3]) {
    // Initialise v to identity
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            v[i][j] = (i == j) ? 1.0 : 0.0;

    for (int iter = 0; iter < 50; ++iter) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        double maxVal = std::abs(a[0][1]);
        if (std::abs(a[0][2]) > maxVal) { p = 0; q = 2; maxVal = std::abs(a[0][2]); }
        if (std::abs(a[1][2]) > maxVal) { p = 1; q = 2; maxVal = std::abs(a[1][2]); }
        if (maxVal < 1e-14) break;

        // Compute rotation angle
        double dif = a[q][q] - a[p][p];
        if (std::abs(a[p][q]) > 1e-300) {
            double theta = dif / (2.0 * a[p][q]);
            double t = 1.0 / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            if (theta < 0.0) t = -t;
            double c = 1.0 / std::sqrt(t * t + 1.0);
            double s = t * c;

            // Update a
            a[p][p] -= t * a[p][q];
            a[q][q] += t * a[p][q];
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for (int r = 0; r < 3; ++r) {
                if (r != p && r != q) {
                    double ap = c * a[r][p] - s * a[r][q];
                    double aq = s * a[r][p] + c * a[r][q];
                    a[r][p] = ap; a[p][r] = ap;
                    a[r][q] = aq; a[q][r] = aq;
                }
            }

            // Update eigenvectors
            for (int r = 0; r < 3; ++r) {
                double vp = c * v[p][r] - s * v[q][r];
                double vq = s * v[p][r] + c * v[q][r];
                v[p][r] = vp;
                v[q][r] = vq;
            }
        }
    }
}

void compute_principal_3d(double sx, double sy, double sz,
                          double sxy, double syz, double szx,
                          double p[3], double v[3][3]) {
    double a[3][3] = {
        {sx,  sxy, szx},
        {sxy, sy,  syz},
        {szx, syz, sz }
    };
    jacobi3(a, v);

    // Extract eigenvalues and sort descending (major → minor)
    p[0] = a[0][0];
    p[1] = a[1][1];
    p[2] = a[2][2];

    // Sort by descending eigenvalue; keep eigenvectors in sync
    // Simple 3-element insertion sort
    for (int i = 1; i < 3; ++i) {
        for (int j = i; j > 0 && p[j] > p[j-1]; --j) {
            std::swap(p[j], p[j-1]);
            for (int k = 0; k < 3; ++k)
                std::swap(v[j][k], v[j-1][k]);
        }
    }
}

// ── F06 writer ────────────────────────────────────────────────────────────────

void F06Writer::write(const SolverResults& results, const Model& model,
                      const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) throw SolverError(std::format("Cannot write F06: {}", path.string()));
    write(results, model, f);
}

void F06Writer::write(const SolverResults& results, const Model& model,
                      std::ostream& out) {
    write_header(out);
    for (const auto& sc : results.subcases) {
        // Find the SubCase from the model for output flags
        bool do_disp   = false;
        bool do_stress = false;
        for (const auto& msc : model.analysis.subcases) {
            if (msc.id == sc.id) {
                do_disp   = msc.disp_print;
                do_stress = msc.stress_print;
                break;
            }
        }

        out << "\n1                                             SUBCASE " << sc.id << "\n";
        if (!sc.label.empty())
            out << "                                              " << sc.label << "\n";

        if (do_disp)   write_displacement_table(sc, out);
        if (do_stress) {
            write_quad4_stress_table(sc, out);
            write_tria3_stress_table(sc, out);
            write_solid_stress_table(sc, out, ElementType::CHEXA8);
            write_solid_stress_table(sc, out, ElementType::CTETRA4);
            write_solid_stress_table(sc, out, ElementType::CTETRA10);
            write_solid_stress_table(sc, out, ElementType::CPENTA6);
        }
    }
    out << "\n\n                     * * * END OF JOB * * *\n\n";
}

void F06Writer::write_header(std::ostream& out) {
    // Date/time for the header
    std::time_t t = std::time(nullptr);
    char date_buf[32];
    std::strftime(date_buf, sizeof(date_buf), "%B %e, %Y", std::localtime(&t));

    out << "1                           N A S T R A N - C O M P A T I B L E   S O L V E R"
        << "                                                          " << date_buf << "\n";
    out << "0                                                                         \n";
    out << "         S O L   1 0 1   L I N E A R   S T A T I C   A N A L Y S I S\n";
    out << "\n";
}

void F06Writer::write_displacement_table(const SubCaseResults& sc,
                                         std::ostream& out) {
    if (sc.displacements.empty()) return;

    out << "\n                                             D I S P L A C E M E N T   V E C T O R\n\n";
    out << "      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3\n";

    for (const auto& nd : sc.displacements) {
        out << std::setw(12) << nd.node.value << "        G   ";
        for (int i = 0; i < 6; ++i)
            out << std::setw(15) << std::setprecision(6) << std::scientific << nd.d[i];
        out << "\n";
    }
}

void F06Writer::write_quad4_stress_table(const SubCaseResults& sc,
                                          std::ostream& out) {
    // Filter CQUAD4 plate stresses
    bool has_quad4 = false;
    for (const auto& ps : sc.plate_stresses)
        if (ps.etype == ElementType::CQUAD4) { has_quad4 = true; break; }
    if (!has_quad4) return;

    out << "\n                     S T R E S S E S   I N   Q U A D R I L A T E R A L   E L E M E N T S   ( C Q U A D 4 )\n\n";
    out << "  ELEMENT-ID    FIBER         NORMAL-X       NORMAL-Y      SHEAR-XY       ANGLE         MAJOR          MINOR        VON MISES\n";

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != ElementType::CQUAD4) continue;
        double major, minor, angle_deg;
        compute_principal_2d(ps.sx, ps.sy, ps.sxy, major, minor, angle_deg);

        out << std::setw(12) << ps.eid.value;
        out << std::setw(15) << std::setprecision(6) << std::scientific << 0.0; // fiber distance
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sxy;
        out << std::setw(10) << std::setprecision(4) << std::fixed    << angle_deg;
        out << std::setw(15) << std::setprecision(6) << std::scientific << major;
        out << std::setw(15) << std::setprecision(6) << std::scientific << minor;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.von_mises;
        out << "\n";
    }
}

void F06Writer::write_tria3_stress_table(const SubCaseResults& sc,
                                          std::ostream& out) {
    bool has_tria3 = false;
    for (const auto& ps : sc.plate_stresses)
        if (ps.etype == ElementType::CTRIA3) { has_tria3 = true; break; }
    if (!has_tria3) return;

    out << "\n                         S T R E S S E S   I N   T R I A N G U L A R   E L E M E N T S   ( C T R I A 3 )\n\n";
    out << "  ELEMENT-ID    FIBER         NORMAL-X       NORMAL-Y      SHEAR-XY       ANGLE         MAJOR          MINOR        VON MISES\n";

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != ElementType::CTRIA3) continue;
        double major, minor, angle_deg;
        compute_principal_2d(ps.sx, ps.sy, ps.sxy, major, minor, angle_deg);

        out << std::setw(12) << ps.eid.value;
        out << std::setw(15) << std::setprecision(6) << std::scientific << 0.0;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.sxy;
        out << std::setw(10) << std::setprecision(4) << std::fixed    << angle_deg;
        out << std::setw(15) << std::setprecision(6) << std::scientific << major;
        out << std::setw(15) << std::setprecision(6) << std::scientific << minor;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ps.von_mises;
        out << "\n";
    }
}

void F06Writer::write_solid_stress_table(const SubCaseResults& sc,
                                          std::ostream& out,
                                          ElementType etype) {
    bool has_type = false;
    for (const auto& ss : sc.solid_stresses)
        if (ss.etype == etype) { has_type = true; break; }
    if (!has_type) return;

    const char* title = "S O L I D";
    int ngrids = 0;
    if (etype == ElementType::CHEXA8) {
        title  = "H E X A H E D R O N   E L E M E N T S   ( C H E X A )";
        ngrids = 8;
    } else if (etype == ElementType::CTETRA4) {
        title  = "T E T R A H E D R O N   E L E M E N T S   ( C T E T R A )";
        ngrids = 4;
    } else if (etype == ElementType::CTETRA10) {
        title  = "T E T R A H E D R O N   E L E M E N T S   ( C T E T R A 1 0 )";
        ngrids = 10;
    } else if (etype == ElementType::CPENTA6) {
        title  = "P E N T A H E D R O N   E L E M E N T S   ( C P E N T A )";
        ngrids = 6;
    }

    out << "\n                         S T R E S S E S   I N   " << title << "\n\n";
    out << "  ELEMENT-ID  GRID-ID    NORMAL-X       NORMAL-Y       NORMAL-Z      SHEAR-XY       SHEAR-YZ       SHEAR-ZX       VON MISES\n";

    for (const auto& ss : sc.solid_stresses) {
        if (ss.etype != etype) continue;
        // Print centroid-only line with CEN/N label
        char cen_label[10];
        std::snprintf(cen_label, sizeof(cen_label), "CEN/%d", ngrids);

        out << std::setw(12) << ss.eid.value;
        out << std::setw(7)  << cen_label;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sz;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.sxy;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.syz;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.szx;
        out << std::setw(15) << std::setprecision(6) << std::scientific << ss.von_mises;
        out << "\n";
    }
}

// ── F06 modal output ──────────────────────────────────────────────────────────

void F06Writer::write_modal(const ModalSolverResults& results, const Model& model,
                            const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) throw SolverError(std::format("Cannot write F06: {}", path.string()));
    write_modal(results, model, f);
}

void F06Writer::write_modal(const ModalSolverResults& results, const Model& /*model*/,
                            std::ostream& out) {
    write_modal_header(out);
    for (const auto& msc : results.subcases) {
        bool do_vec = msc.eigvec_print;

        out << "\n1                                             SUBCASE " << msc.id << "\n";
        if (!msc.label.empty())
            out << "                                              " << msc.label << "\n";

        write_eigenvalue_table(msc, out);

        if (do_vec)
            for (const auto& mode : msc.modes)
                write_eigenvector_table(mode, msc.label, out);
    }
    out << "\n\n                     * * * END OF JOB * * *\n\n";
}

void F06Writer::write_modal_header(std::ostream& out) {
    std::time_t t = std::time(nullptr);
    char date_buf[32];
    std::strftime(date_buf, sizeof(date_buf), "%B %e, %Y", std::localtime(&t));

    out << "1                           N A S T R A N - C O M P A T I B L E   S O L V E R"
        << "                                                          " << date_buf << "\n";
    out << "0                                                                         \n";
    out << "         S O L   1 0 3   N O R M A L   M O D E S   A N A L Y S I S\n";
    out << "\n";
}

void F06Writer::write_eigenvalue_table(const ModalSubCaseResults& msc,
                                       std::ostream& out) {
    if (msc.modes.empty()) return;

    out << "\n                                            R E A L   E I G E N V A L U E S\n";
    out << "   MODE  EXTRACTION      EIGENVALUE           RADIANS              CYCLES            GENERALIZED         GENERALIZED        \n";
    out << "  NUMBER   ORDER                                                                        MASS              STIFFNESS\n\n";

    for (const auto& mode : msc.modes) {
        double gen_stiff = mode.eigenvalue * mode.gen_mass;
        out << std::setw(9)  << mode.mode_number;
        out << std::setw(8)  << mode.mode_number; // extraction order = mode number
        out << std::uppercase;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.eigenvalue;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.radians_per_sec;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.cycles_per_sec;
        out << std::setw(20) << std::setprecision(6) << std::scientific << mode.gen_mass;
        out << std::setw(20) << std::setprecision(6) << std::scientific << gen_stiff;
        out << std::nouppercase;
        out << "\n";
    }
}

void F06Writer::write_eigenvector_table(const ModeResult& mode,
                                        const std::string& label,
                                        std::ostream& out) {
    out << "\nOUTPUT FOR EIGENVECTOR" << std::setw(9) << mode.mode_number << "\n";
    out << label << "\n";
    out << "\n                                                         E I G E N V E C T O R\n";
    out << "                                              (in global coordinate system at each grid)\n";
    out << "           GRID     COORD      T1            T2            T3            R1            R2            R3\n";
    out << "                     SYS\n";

    for (const auto& nd : mode.shape) {
        out << std::setw(15) << nd.node.value;
        out << std::setw(9)  << 0; // global coord sys
        out << std::uppercase;
        for (int i = 0; i < 6; ++i) {
            double v = nd.d[i];
            if (v == 0.0) {
                out << "  0.0         ";
            } else {
                out << std::setw(14) << std::setprecision(6) << std::scientific << v;
            }
        }
        out << std::nouppercase;
        out << "\n";
    }
}

} // namespace vibetran
