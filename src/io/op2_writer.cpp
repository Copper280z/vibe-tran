// src/io/op2_writer.cpp
// MSC Nastran OP2 binary writer, little-endian, post=-1 format.
//
// Format reference: pyNastran op2_writer.py / table_object.py / oes_*.py
//
// Record encoding: Fortran unformatted sequential I/O
//   [int32 nbytes] [nbytes of data] [int32 nbytes]
//
// File structure:
//   File header (MSC post=-1 tape-code block)
//   For each result table (OUGV1, OES1X per element-type):
//     Table header:  [4,2,4][8,name8,8][4,-1,4][4,7,4][28,magic7,28]
//                    [4,-2,4][4,1,4][4,0,4][4,7,4][28,subtable_date,28]
//     For each subcase (static → one time-step):
//       Table-3 record (584-byte metadata header)
//       Table-4 record (float32 result data)
//     Inter-result footer markers
//   Table close: [4,0,4]
//   File close:  [4,0,4]

#include "io/op2_writer.hpp"
#include <cstring>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vibetran {

// ── Primitive I/O helpers ─────────────────────────────────────────────────────

static void write_i32(std::ostream& f, int32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}

static void write_f32(std::ostream& f, float v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}

/// Write a Fortran unformatted record: [nbytes][data...][nbytes]
static void write_record(std::ostream& f, const void* data, int32_t nbytes) {
    write_i32(f, nbytes);
    f.write(reinterpret_cast<const char*>(data), nbytes);
    write_i32(f, nbytes);
}

/// Write a single int32 as a Fortran record of 4 bytes.
static void write_record_i32(std::ostream& f, int32_t v) {
    write_record(f, &v, 4);
}

/// Write a series of int32 Fortran-record markers: [4,v,4] per value.
template<typename... Vs>
static void write_markers(std::ostream& f, Vs... vs) {
    int32_t vals[] = {static_cast<int32_t>(vs)...};
    for (int32_t v : vals) write_record_i32(f, v);
}

// ── OP2 table header ──────────────────────────────────────────────────────────
// Writes: [4,2,4][8,name8,8]
//         [4,-1,4][4,7,4]
//         [28, 102,0,0,0,512,0,0, 28]
//         [4,-2,4][4,1,4][4,0,4]
//         [4,7,4]
//         [28, subtable(8), month,day,dyear,0,1, 28]

static void write_table_header(std::ostream& f, const char name8[8],
                                int month, int day, int dyear) {
    // [4][2][4] [8][name8][8]
    {
        int32_t m[3] = {4, 2, 4};
        f.write(reinterpret_cast<const char*>(m), 12);
        int32_t r = 8;
        f.write(reinterpret_cast<const char*>(&r), 4);
        f.write(name8, 8);
        f.write(reinterpret_cast<const char*>(&r), 4);
    }
    // [4,-1,4] [4,7,4]
    {
        int32_t m1[3] = {4, -1, 4};
        int32_t m2[3] = {4,  7, 4};
        f.write(reinterpret_cast<const char*>(m1), 12);
        f.write(reinterpret_cast<const char*>(m2), 12);
    }
    // [28][102,0,0,0,512,0,0][28]
    {
        int32_t rec[9] = {28, 102, 0, 0, 0, 512, 0, 0, 28};
        f.write(reinterpret_cast<const char*>(rec), 36);
    }
    // [4,-2,4][4,1,4][4,0,4]
    {
        int32_t m[9] = {4,-2,4, 4,1,4, 4,0,4};
        f.write(reinterpret_cast<const char*>(m), 36);
    }
    // [4,7,4][28][subtable(8)][month,day,dyear,0,1][28]
    // subtable = b'OUG1    ' for all tables (pyNastran default)
    {
        int32_t m3[3] = {4, 7, 4};
        f.write(reinterpret_cast<const char*>(m3), 12);
        int32_t r = 28;
        f.write(reinterpret_cast<const char*>(&r), 4);
        const char subtable[8] = {'O','U','G','1',' ',' ',' ',' '};
        f.write(subtable, 8);
        int32_t date[5] = {month, day, dyear, 0, 1};
        f.write(reinterpret_cast<const char*>(date), 20);
        f.write(reinterpret_cast<const char*>(&r), 4);
    }
}

// ── TABLE-3 record (584-byte subcase metadata) ────────────────────────────────

static void write_table3(std::ostream& f, bool new_result, int itable,
                         int approach_code, int table_code,
                         int element_type, int isubcase,
                         int num_wide,
                         const std::string& title,
                         const std::string& subtitle,
                         const std::string& label) {
    // Prefix markers before the 584-byte record
    if (new_result && itable != -3) {
        int32_t m[3] = {4, 146, 4};
        f.write(reinterpret_cast<const char*>(m), 12);
    } else {
        int32_t m[12] = {4, itable, 4, 4, 1, 4, 4, 0, 4, 4, 146, 4};
        f.write(reinterpret_cast<const char*>(m), 48);
    }

    // Build the 584-byte payload: 50 × int32 + 3 × 128-char strings
    static constexpr int N_INTS = 50;
    static constexpr int STR_LEN = 128;
    int32_t ints[N_INTS] = {};
    ints[0]  = approach_code;   // aCode
    ints[1]  = table_code;      // tCode
    ints[2]  = element_type;    // element_type (0 for OUGV1)
    ints[3]  = isubcase;        // isubcase
    ints[4]  = 0;               // lsdvmn (load set id, 0 for static)
    ints[7]  = 0;               // random_code
    ints[8]  = 1;               // format_code = real
    ints[9]  = num_wide;        // num_wide
    ints[10] = 0;               // s_code / oCode (0 = stress, von Mises)
    ints[22] = 0;               // thermal

    char strings[3][STR_LEN];
    auto fill_str = [&](char dst[STR_LEN], const std::string& src) {
        std::memset(dst, ' ', STR_LEN);
        size_t n = std::min(src.size(), (size_t)STR_LEN);
        std::memcpy(dst, src.data(), n);
    };
    fill_str(strings[0], title);
    fill_str(strings[1], subtitle);
    fill_str(strings[2], label);

    int32_t rec_len = N_INTS * 4 + 3 * STR_LEN; // = 584
    f.write(reinterpret_cast<const char*>(&rec_len), 4);
    f.write(reinterpret_cast<const char*>(ints), N_INTS * 4);
    f.write(strings[0], STR_LEN);
    f.write(strings[1], STR_LEN);
    f.write(strings[2], STR_LEN);
    f.write(reinterpret_cast<const char*>(&rec_len), 4);
}

// ── File header (MSC post=-1) ─────────────────────────────────────────────────

static void write_file_header(std::ostream& f, int month, int day, int dyear) {
    // [4,3,4]
    {
        int32_t m[3] = {4, 3, 4};
        f.write(reinterpret_cast<const char*>(m), 12);
    }
    // [12, day, month, dyear, 12][4,7,4][28, tape_code, 28]
    {
        int32_t date_rec[5] = {12, day, month, dyear, 12};
        f.write(reinterpret_cast<const char*>(date_rec), 20);

        int32_t m2[3] = {4, 7, 4};
        f.write(reinterpret_cast<const char*>(m2), 12);

        const char tape[] = "NASTRAN FORT TAPE ID CODE - ";  // 28 chars
        int32_t r = 28;
        f.write(reinterpret_cast<const char*>(&r), 4);
        f.write(tape, 28);
        f.write(reinterpret_cast<const char*>(&r), 4);
    }
    // [4,2,4][8, 'XXXXXXXX', 8]
    {
        int32_t m3[3] = {4, 2, 4};
        f.write(reinterpret_cast<const char*>(m3), 12);
        int32_t r = 8;
        f.write(reinterpret_cast<const char*>(&r), 4);
        const char ver[] = "XXXXXXXX";
        f.write(ver, 8);
        f.write(reinterpret_cast<const char*>(&r), 4);
    }
    // [4,-1,4][4,0,4]
    {
        int32_t m4[6] = {4,-1,4, 4,0,4};
        f.write(reinterpret_cast<const char*>(m4), 24);
    }
}

// ── int32 viewed as float32 helper ───────────────────────────────────────────

static float i32_as_f32(int32_t v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

// ── Write one displacement TABLE-4 data record ───────────────────────────────
// OUGV1: each node → [node_id*10+2, grid_type, T1..R3] as float32 (8 words)

static int write_ougv1(std::ostream& f,
                       const SubCaseResults& sc,
                       bool new_result, int itable,
                       const std::string& subtitle) {
    if (sc.displacements.empty()) return itable;

    const int approach_code = 12; // static=1, plot=2 → 1*10+2
    const int table_code    =  1; // OUGV1
    const int num_wide      =  8; // 2 ints + 6 floats per node
    const int nn = static_cast<int>(sc.displacements.size());
    const int ntotal = nn * num_wide; // total 32-bit words in data record

    // TABLE-3
    write_table3(f, new_result, itable,
                 approach_code, table_code,
                 /*element_type=*/0, sc.id,
                 num_wide,
                 /*title=*/"", subtitle, sc.label);
    itable--;

    // TABLE-4 header markers
    {
        int32_t m[13] = {4, itable, 4,
                         4, 1, 4,
                         4, 0, 4,
                         4, ntotal, 4,
                         4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    // TABLE-4 data: [node_id*10+2 (as f32), grid_type (as f32), T1..R3 (f32)] per node
    for (const auto& nd : sc.displacements) {
        int32_t id_dev = static_cast<int32_t>(nd.node.value) * 10 + 2;
        write_f32(f, i32_as_f32(id_dev));
        write_f32(f, i32_as_f32(1));   // grid_type G=1
        for (int i = 0; i < 6; ++i)
            write_f32(f, static_cast<float>(nd.d[i]));
    }

    // TABLE-4 trailing length marker
    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// ── Write one OES1X plate stress TABLE-4 ─────────────────────────────────────
// CQUAD4 (element_type=33) or CTRIA3 (element_type=74), centroid only.
// num_wide=17 per element:
//   [eid_device, fd1, sx1,sy1,txy1,angle1,major1,minor1,vm1,
//                fd2, sx2,sy2,txy2,angle2,major2,minor2,vm2]

static int write_oes1x_plate(std::ostream& f,
                              const SubCaseResults& sc,
                              bool new_result, int itable,
                              ElementType etype,
                              const std::string& subtitle) {
    // Count matching elements
    int nel = 0;
    for (const auto& ps : sc.plate_stresses)
        if (ps.etype == etype) ++nel;
    if (nel == 0) return itable;

    const int approach_code = 12;
    const int table_code    =  5; // OES1X
    const int num_wide      = 17;
    const int etype_id = (etype == ElementType::CQUAD4) ? 33 : 74;
    const int ntotal = nel * num_wide;

    write_table3(f, new_result, itable,
                 approach_code, table_code,
                 etype_id, sc.id,
                 num_wide,
                 "", subtitle, sc.label);
    itable--;

    {
        int32_t m[13] = {4, itable, 4,
                         4, 1, 4,
                         4, 0, 4,
                         4, ntotal, 4,
                         4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    for (const auto& ps : sc.plate_stresses) {
        if (ps.etype != etype) continue;
        double major, minor, angle_deg;
        compute_principal_2d(ps.sx, ps.sy, ps.sxy, major, minor, angle_deg);

        int32_t eid_dev = static_cast<int32_t>(ps.eid.value) * 10 + 2;
        // Layer 1 (fiber_dist = 0, same membrane stress at both layers)
        write_f32(f, i32_as_f32(eid_dev));
        write_f32(f, 0.0f);  // fiber dist
        write_f32(f, static_cast<float>(ps.sx));
        write_f32(f, static_cast<float>(ps.sy));
        write_f32(f, static_cast<float>(ps.sxy));
        write_f32(f, static_cast<float>(angle_deg));
        write_f32(f, static_cast<float>(major));
        write_f32(f, static_cast<float>(minor));
        write_f32(f, static_cast<float>(ps.von_mises));
        // Layer 2 (same as layer 1 for membrane-only)
        write_f32(f, 0.0f);
        write_f32(f, static_cast<float>(ps.sx));
        write_f32(f, static_cast<float>(ps.sy));
        write_f32(f, static_cast<float>(ps.sxy));
        write_f32(f, static_cast<float>(angle_deg));
        write_f32(f, static_cast<float>(major));
        write_f32(f, static_cast<float>(minor));
        write_f32(f, static_cast<float>(ps.von_mises));
    }

    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// ── Write one OES1X solid stress TABLE-4 ─────────────────────────────────────
// CTETRA4 (element_type=39, num_wide=109) or CHEXA8 (element_type=67, num_wide=193).
// We have centroid data only; corner nodes are filled with the centroid values.
//
// Header (4 words per element): [eid_device, cid=0, 'CEN/', ngrids]
// Per node (21 words):
//   [node_id, oxx, txy, o1, dir1y, dir1z, dir1x, p, vm,
//             oyy, tyz, o2, dir2y, dir2z, dir2x,
//             ozz, txz, o3, dir3y, dir3z, dir3x]

static int write_oes1x_solid(std::ostream& f,
                              const SubCaseResults& sc,
                              bool new_result, int itable,
                              ElementType etype,
                              const std::string& subtitle) {
    int nel = 0;
    for (const auto& ss : sc.solid_stresses)
        if (ss.etype == etype) ++nel;
    if (nel == 0) return itable;

    int etype_id, ngrids;
    if (etype == ElementType::CTETRA4) {
        etype_id = 39;
        ngrids   = 4;
    } else if (etype == ElementType::CTETRA10) {
        etype_id = 99;
        ngrids   = 10;
    } else if (etype == ElementType::CPENTA6) {
        etype_id = 91;
        ngrids   = 6;
    } else {
        etype_id = 67;  // CHEXA8
        ngrids   = 8;
    }

    // nnodes_per_element = ngrids + 1 (centroid)
    const int nnodes_per_elem = ngrids + 1;
    const int num_wide        = 4 + 21 * nnodes_per_elem;
    const int ntotal          = nel * num_wide;

    const int approach_code = 12;
    const int table_code    =  5; // OES1X

    write_table3(f, new_result, itable,
                 approach_code, table_code,
                 etype_id, sc.id,
                 num_wide,
                 "", subtitle, sc.label);
    itable--;

    {
        int32_t m[13] = {4, itable, 4,
                         4, 1, 4,
                         4, 0, 4,
                         4, ntotal, 4,
                         4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    // 'CEN/' as 4 bytes interpreted as int32
    int32_t cen_bytes;
    {
        const char cen[] = {'C','E','N','/'};
        std::memcpy(&cen_bytes, cen, 4);
    }

    for (const auto& ss : sc.solid_stresses) {
        if (ss.etype != etype) continue;

        // Compute principal stresses and direction cosines
        double p[3], v[3][3];
        compute_principal_3d(ss.sx, ss.sy, ss.sz,
                             ss.sxy, ss.syz, ss.szx,
                             p, v);
        double pressure = -(p[0] + p[1] + p[2]) / 3.0;

        // Element header: [eid_device, cid=0, 'CEN/', ngrids]
        int32_t eid_dev = static_cast<int32_t>(ss.eid.value) * 10 + 2;
        write_f32(f, i32_as_f32(eid_dev));
        write_f32(f, i32_as_f32(0));        // cid
        write_f32(f, i32_as_f32(cen_bytes)); // 'CEN/'
        write_f32(f, i32_as_f32(ngrids));    // number of corner nodes

        // Write nnodes_per_elem node blocks (centroid + corner nodes).
        // Since we only have centroid data, all node blocks use the same values.
        // Centroid node_id = 0; corner node_ids = 1..ngrids (placeholder).
        auto write_node_block = [&](int32_t node_id) {
            // [node_id, oxx, txy, o1, dir1y, dir1z, dir1x, p, vm,
            //           oyy, tyz, o2, dir2y, dir2z, dir2x,
            //           ozz, txz, o3, dir3y, dir3z, dir3x]
            write_f32(f, i32_as_f32(node_id));
            write_f32(f, static_cast<float>(ss.sx));
            write_f32(f, static_cast<float>(ss.sxy));
            write_f32(f, static_cast<float>(p[0]));   // o1
            write_f32(f, static_cast<float>(v[0][1])); // dir1y
            write_f32(f, static_cast<float>(v[0][2])); // dir1z
            write_f32(f, static_cast<float>(v[0][0])); // dir1x
            write_f32(f, static_cast<float>(pressure));
            write_f32(f, static_cast<float>(ss.von_mises));

            write_f32(f, static_cast<float>(ss.sy));
            write_f32(f, static_cast<float>(ss.syz));
            write_f32(f, static_cast<float>(p[1]));   // o2
            write_f32(f, static_cast<float>(v[1][1]));
            write_f32(f, static_cast<float>(v[1][2]));
            write_f32(f, static_cast<float>(v[1][0]));

            write_f32(f, static_cast<float>(ss.sz));
            write_f32(f, static_cast<float>(ss.szx));
            write_f32(f, static_cast<float>(p[2]));   // o3
            write_f32(f, static_cast<float>(v[2][1]));
            write_f32(f, static_cast<float>(v[2][2]));
            write_f32(f, static_cast<float>(v[2][0]));
        };

        write_node_block(0); // centroid
        for (int n = 1; n <= ngrids; ++n)
            write_node_block(static_cast<int32_t>(n)); // corner nodes (placeholder IDs)
    }

    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    itable--;
    return itable;
}

// LAMA table removed — eigenvalue data is embedded in each OUGV1 TABLE-3 record.

// ── Write OUGV1 eigenvector records (one full table per mode, mystran format) ──
// approach_code=21: analysis_code=2 (real modes), device_code=1 (print).
// TABLE-3 ints[5] = eigenvalue as float32 bitcast, ints[6] = radians as float32 bitcast.
// One complete table (header + TABLE-3/4 pair) is written per mode, matching the
// mystran output format expected by community tools (pyNastran, Femap, etc.).

static void write_ougv1_modal_mode(std::ostream& f,
                                   const ModalSubCaseResults& msc,
                                   const ModeResult& mr,
                                   int month, int day, int dyear) {
    if (mr.shape.empty()) return;

    const int approach_code = 21; // analysis_code=2 (real modes), device_code=1
    const int table_code    =  7; // OUGV1 eigenvectors
    const int num_wide      =  8;
    const int nn     = static_cast<int>(mr.shape.size());
    const int ntotal = nn * num_wide;

    // Write fresh table header for this mode
    const char name8[8] = {'O','U','G','V','1',' ',' ',' '};
    write_table_header(f, name8, month, day, dyear);

    // TABLE-3 prefix: [-3, 1, 0, 146]
    {
        int32_t m[12] = {4, -3, 4, 4, 1, 4, 4, 0, 4, 4, 146, 4};
        f.write(reinterpret_cast<const char*>(m), 48);
    }

    // Build TABLE-3 payload (584 bytes = 50 ints + 3×128 chars)
    static constexpr int N_INTS  = 50;
    static constexpr int STR_LEN = 128;
    int32_t ints[N_INTS] = {};
    ints[0] = approach_code;
    ints[1] = table_code;
    ints[2] = 0;            // element_type
    ints[3] = msc.id;       // isubcase
    ints[4] = mr.mode_number; // lsdvmn = mode number
    // ints[5] = eigenvalue as float32 bitcast to int32
    {
        float ev = static_cast<float>(mr.eigenvalue);
        std::memcpy(&ints[5], &ev, 4);
    }
    // ints[6] = radians/sec as float32 bitcast to int32
    {
        float rad = static_cast<float>(mr.radians_per_sec);
        std::memcpy(&ints[6], &rad, 4);
    }
    ints[8] = 1;          // format_code = real
    ints[9] = num_wide;

    char strings[3][STR_LEN];
    auto fill_str = [&](char dst[STR_LEN], const std::string& src) {
        std::memset(dst, ' ', STR_LEN);
        size_t n = std::min(src.size(), (size_t)STR_LEN);
        std::memcpy(dst, src.data(), n);
    };
    fill_str(strings[0], "");
    fill_str(strings[1], "");
    fill_str(strings[2], msc.label);

    int32_t rec_len = N_INTS * 4 + 3 * STR_LEN; // 584
    f.write(reinterpret_cast<const char*>(&rec_len), 4);
    f.write(reinterpret_cast<const char*>(ints), N_INTS * 4);
    f.write(strings[0], STR_LEN);
    f.write(strings[1], STR_LEN);
    f.write(strings[2], STR_LEN);
    f.write(reinterpret_cast<const char*>(&rec_len), 4);

    // TABLE-4 header markers: [-4, 1, 0, ntotal]
    {
        int32_t m[13] = {4, -4, 4,
                         4, 1, 4,
                         4, 0, 4,
                         4, ntotal, 4,
                         4 * ntotal};
        f.write(reinterpret_cast<const char*>(m), 52);
    }

    // TABLE-4 data: one record per node [node_id*10+2 (as f32), grid_type=1, T1..R3]
    for (const auto& nd : mr.shape) {
        int32_t id_dev = static_cast<int32_t>(nd.node.value) * 10 + 2;
        write_f32(f, i32_as_f32(id_dev));
        write_f32(f, i32_as_f32(1)); // grid type G=1
        for (int i = 0; i < 6; ++i)
            write_f32(f, static_cast<float>(nd.d[i]));
    }

    // TABLE-4 trailing length marker + footer
    {
        int32_t tail = 4 * ntotal;
        f.write(reinterpret_cast<const char*>(&tail), 4);
    }
    // Footer: [-5, 1, 0] then table close [0]
    {
        int32_t footer[9] = {4, -5, 4, 4, 1, 4, 4, 0, 4};
        f.write(reinterpret_cast<const char*>(footer), 36);
    }
    write_markers(f, 0); // table close
}

// ── Op2Writer::write_modal ────────────────────────────────────────────────────

void Op2Writer::write_modal(const ModalSolverResults& results, const Model& model,
                            const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw SolverError(std::format("Cannot write OP2: {}", path.string()));

    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);
    int month = lt->tm_mon + 1;
    int day   = lt->tm_mday;
    int dyear = lt->tm_year - 100;

    write_file_header(f, month, day, dyear);

    // ── OUGV1 (eigenvectors) ──────────────────────────────────────────────────
    // One complete OUGV1 table is written per mode, matching the mystran format.
    // eigvec_plot flag is taken from ModalSubCaseResults (which already applies
    // the DISPLACEMENT→EIGENVECTOR alias performed in the modal solver).
    for (const auto& msc : results.subcases) {
        if (!msc.eigvec_plot || msc.modes.empty()) continue;
        for (const auto& mr : msc.modes)
            write_ougv1_modal_mode(f, msc, mr, month, day, dyear);
    }

    (void)model;

    write_markers(f, 0); // file end
}

// ── Op2Writer::write ──────────────────────────────────────────────────────────

void Op2Writer::write(const SolverResults& results, const Model& model,
                      const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw SolverError(std::format("Cannot write OP2: {}", path.string()));

    // Date for header
    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);
    int month = lt->tm_mon + 1;
    int day   = lt->tm_mday;
    int dyear = lt->tm_year - 100; // years since 2000

    write_file_header(f, month, day, dyear);

    // Collect output flags per subcase
    auto get_flags = [&](int sc_id, bool& do_disp, bool& do_stress) {
        do_disp = do_stress = false;
        for (const auto& msc : model.analysis.subcases) {
            if (msc.id == sc_id) {
                do_disp   = msc.disp_print   || msc.disp_plot;
                do_stress = msc.stress_print || msc.stress_plot;
                return;
            }
        }
    };

    // ── OUGV1 table (displacements) ───────────────────────────────────────────
    {
        bool any_disp = false;
        for (const auto& sc : results.subcases) {
            bool dd, ds;
            get_flags(sc.id, dd, ds);
            if (dd && !sc.displacements.empty()) { any_disp = true; break; }
        }
        if (any_disp) {
            const char name8[8] = {'O','U','G','V','1',' ',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -1;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                bool dd, ds;
                get_flags(sc.id, dd, ds);
                if (!dd || sc.displacements.empty()) continue;
                if (itable == -1) { itable = -3; }
                itable = write_ougv1(f, sc, new_result, itable, sc.label);
                new_result = false;
                // Inter-result footer
                int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            // Table close
            write_markers(f, 0);
        }
    }

    // ── OES1X table for CQUAD4 ────────────────────────────────────────────────
    {
        bool any = false;
        for (const auto& sc : results.subcases) {
            bool dd, ds;
            get_flags(sc.id, dd, ds);
            if (!ds) continue;
            for (const auto& ps : sc.plate_stresses)
                if (ps.etype == ElementType::CQUAD4) { any = true; break; }
            if (any) break;
        }
        if (any) {
            const char name8[8] = {'O','E','S','1','X',' ',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -3;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                bool dd, ds;
                get_flags(sc.id, dd, ds);
                if (!ds) continue;
                itable = write_oes1x_plate(f, sc, new_result, itable,
                                           ElementType::CQUAD4, sc.label);
                if (!new_result || itable < -3) new_result = false;
                int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            write_markers(f, 0);
        }
    }

    // ── OES1X table for CTRIA3 ────────────────────────────────────────────────
    {
        bool any = false;
        for (const auto& sc : results.subcases) {
            bool dd, ds;
            get_flags(sc.id, dd, ds);
            if (!ds) continue;
            for (const auto& ps : sc.plate_stresses)
                if (ps.etype == ElementType::CTRIA3) { any = true; break; }
            if (any) break;
        }
        if (any) {
            const char name8[8] = {'O','E','S','1','X',' ',' ',' '};
            write_table_header(f, name8, month, day, dyear);

            int itable = -3;
            bool new_result = true;
            for (const auto& sc : results.subcases) {
                bool dd, ds;
                get_flags(sc.id, dd, ds);
                if (!ds) continue;
                itable = write_oes1x_plate(f, sc, new_result, itable,
                                           ElementType::CTRIA3, sc.label);
                if (!new_result || itable < -3) new_result = false;
                int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
                f.write(reinterpret_cast<const char*>(footer), 36);
            }
            write_markers(f, 0);
        }
    }

    // ── OES1X tables for solid elements ──────────────────────────────────────
    for (ElementType etype : {ElementType::CTETRA4, ElementType::CTETRA10,
                              ElementType::CHEXA8, ElementType::CPENTA6}) {
        bool any = false;
        for (const auto& sc : results.subcases) {
            bool dd, ds;
            get_flags(sc.id, dd, ds);
            if (!ds) continue;
            for (const auto& ss : sc.solid_stresses)
                if (ss.etype == etype) { any = true; break; }
            if (any) break;
        }
        if (!any) continue;

        const char name8[8] = {'O','E','S','1','X',' ',' ',' '};
        write_table_header(f, name8, month, day, dyear);

        int itable = -3;
        bool new_result = true;
        for (const auto& sc : results.subcases) {
            bool dd, ds;
            get_flags(sc.id, dd, ds);
            if (!ds) continue;
            itable = write_oes1x_solid(f, sc, new_result, itable,
                                       etype, sc.label);
            if (!new_result || itable < -3) new_result = false;
            int32_t footer[9] = {4, itable, 4, 4, 1, 4, 4, 0, 4};
            f.write(reinterpret_cast<const char*>(footer), 36);
        }
        write_markers(f, 0);
    }

    // File end marker
    write_markers(f, 0);
}

} // namespace vibetran
