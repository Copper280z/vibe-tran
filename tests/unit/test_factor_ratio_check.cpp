#include <gtest/gtest.h>

#include "core/factor_ratio_check.hpp"

#include <stdexcept>
#include <vector>

using namespace vibestran;

TEST(FactorRatioCheckPolicy, DefaultsAreStrictAndFatal) {
    Model model;
    const FactorRatioCheckPolicy policy =
        build_factor_ratio_check_policy(model, "[subcase 1]", "stiffness matrix");

    EXPECT_DOUBLE_EQ(policy.maxratio, 1.0e7);
    EXPECT_TRUE(policy.fatal);
    EXPECT_EQ(policy.context, "[subcase 1]");
    EXPECT_EQ(policy.matrix_name, "stiffness matrix");
}

TEST(FactorRatioCheckPolicy, LenientModeWithoutExplicitBailoutWarns) {
    Model model;
    model.params["CHECKMODE"] = "LENIENT";

    const FactorRatioCheckPolicy policy =
        build_factor_ratio_check_policy(model, "[modal sc 1]", "shift matrix");

    EXPECT_DOUBLE_EQ(policy.maxratio, 1.0e7);
    EXPECT_FALSE(policy.fatal);
    EXPECT_EQ(policy.context, "[modal sc 1]");
    EXPECT_EQ(policy.matrix_name, "shift matrix");
}

TEST(FactorRatioCheckPolicy, StrictNegativeBailoutWarns) {
    Model model;
    model.params["BAILOUT"] = "-1";

    const FactorRatioCheckPolicy policy =
        build_factor_ratio_check_policy(model, "[subcase 2]", "stiffness matrix");

    EXPECT_FALSE(policy.fatal);
}

TEST(FactorRatioCheck, ReportsExceededRatio) {
    const std::vector<double> matrix_diag{100.0, 2.0, 3.0};
    const std::vector<double> factor_diag{1.0, 1.0, 2.0};

    const FactorRatioCheckResult result =
        evaluate_factor_ratio_check(matrix_diag, factor_diag, 50.0);

    EXPECT_EQ(result.status, FactorRatioCheckStatus::RatioExceeded);
    EXPECT_DOUBLE_EQ(result.max_ratio, 100.0);
    EXPECT_EQ(result.row_1based, 1);
    EXPECT_DOUBLE_EQ(result.matrix_diag, 100.0);
    EXPECT_DOUBLE_EQ(result.factor_diag, 1.0);
}

TEST(FactorRatioCheck, ReportsNonPositiveFactorDiagonal) {
    const std::vector<double> matrix_diag{10.0, 20.0, 30.0};
    const std::vector<double> factor_diag{1.0, 0.0, 5.0};

    const FactorRatioCheckResult result =
        evaluate_factor_ratio_check(matrix_diag, factor_diag, 1.0e7);

    EXPECT_EQ(result.status, FactorRatioCheckStatus::NonPositiveFactorDiagonal);
    EXPECT_EQ(result.row_1based, 2);
    EXPECT_DOUBLE_EQ(result.matrix_diag, 20.0);
    EXPECT_DOUBLE_EQ(result.factor_diag, 0.0);
}

TEST(FactorRatioCheck, RejectsMismatchedDiagonalLengths) {
    const std::vector<double> matrix_diag{1.0, 2.0};
    const std::vector<double> factor_diag{1.0};

    EXPECT_THROW(
        evaluate_factor_ratio_check(matrix_diag, factor_diag, 1.0e7),
        std::invalid_argument);
}
