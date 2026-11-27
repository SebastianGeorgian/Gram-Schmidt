import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Random;

/** Dependency-free unit tests; run with {@code ./build.sh test}. */
public final class GramSchmidtTest {

    private static int checks, failures;

    public static void main(String[] args) throws Exception {
        independentColumns();
        dependentColumns();
        textbookExample();
        tallMatrix();
        zeroAndDegenerate();
        illConditioned();
        randomMatrices();
        parsing();
        commandLine();
        if (failures > 0) {
            System.err.printf("%d of %d checks FAILED%n", failures, checks);
            System.exit(1);
        }
        System.out.printf("All %d checks passed%n", checks);
    }

    private static void independentColumns() {
        double[][] a = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
        var r = GramSchmidt.decompose(a);
        check(r.rank() == 3, "full rank");
        check(r.dependent().isEmpty(), "no dependent columns");
        valid(a, r, "3x3 independent");
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < i; j++)
                check(Math.abs(r.r()[i][j]) < 1e-15, "R is upper triangular");
        for (int i = 0; i < 3; i++)
            check(r.r()[i][i] > 0, "R has a positive diagonal");
    }

    private static void dependentColumns() {
        // the example from the original version of this repository: column 3 = 2*column 2 - column 1
        double[][] a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        var r = GramSchmidt.decompose(a);
        check(r.rank() == 2, "1..9 matrix has rank 2");
        check(r.dependent().equals(List.of(2)), "third column is dependent");
        valid(a, r, "rank-deficient");

        double[][] repeated = {{1, 1, 2}, {2, 2, 0}, {3, 3, 1}};
        var r2 = GramSchmidt.decompose(repeated);
        check(r2.dependent().equals(List.of(1)), "duplicated column detected");
        valid(repeated, r2, "duplicated column");
    }

    private static void textbookExample() {
        // v1 = (3, 1), v2 = (2, 2): e1 = (3, 1)/sqrt(10), e2 = (-1, 3)/sqrt(10)
        double[][] a = {{3, 2}, {1, 2}};
        double[][] q = GramSchmidt.orthonormalize(a);
        double s = Math.sqrt(10);
        check(near(q[0][0], 3 / s) && near(q[1][0], 1 / s), "first basis vector");
        check(near(q[0][1], -1 / s) && near(q[1][1], 3 / s), "second basis vector");
    }

    private static void tallMatrix() {
        double[][] a = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
        var r = GramSchmidt.decompose(a);
        check(r.q().length == 4 && r.rank() == 2, "4x2 gives a 4x2 Q");
        valid(a, r, "tall matrix");
    }

    private static void zeroAndDegenerate() {
        var zero = GramSchmidt.decompose(new double[][]{{0, 0}, {0, 0}});
        check(zero.rank() == 0 && zero.dependent().equals(List.of(0, 1)), "zero matrix has rank 0");
        var withZero = GramSchmidt.decompose(new double[][]{{0, 1}, {0, 1}});
        check(withZero.rank() == 1 && withZero.dependent().equals(List.of(0)), "zero column skipped");
        var wide = GramSchmidt.decompose(new double[][]{{1, 2, 3}, {4, 5, 7}});
        check(wide.rank() == 2, "2x3 matrix has at most rank 2");
        valid(new double[][]{{1, 2, 3}, {4, 5, 7}}, wide, "wide matrix");
        rejects(() -> GramSchmidt.decompose(new double[0][0]), "empty matrix");
        rejects(() -> GramSchmidt.decompose(new double[][]{{1, 2}, {3}}), "ragged matrix");
        rejects(() -> GramSchmidt.decompose(new double[][]{{1, Double.NaN}}), "NaN entry");
    }

    private static void illConditioned() {
        // Hilbert matrix: classical Gram-Schmidt loses orthogonality badly here
        int n = 10;
        double[][] h = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                h[i][j] = 1.0 / (i + j + 1);
        var r = GramSchmidt.decompose(h, 1e-14);
        check(GramSchmidt.orthogonalityError(r.q()) < 1e-12, "Hilbert(10): Q stays orthonormal");
        check(GramSchmidt.reconstructionError(h, r) < 1e-12, "Hilbert(10): QR = A");
    }

    private static void randomMatrices() {
        var random = new Random(42);
        for (int t = 0; t < 50; t++) {
            int m = 2 + random.nextInt(8), n = 1 + random.nextInt(m);
            double[][] a = new double[m][n];
            for (double[] row : a)
                for (int j = 0; j < n; j++)
                    row[j] = random.nextGaussian() * 100;
            var r = GramSchmidt.decompose(a);
            check(r.rank() == n, "random matrix has full column rank");
            valid(a, r, "random " + m + "x" + n);
        }
    }

    private static void parsing() {
        double[][] a = GramSchmidt.parseMatrix(List.of("# comment", "1 2, 3", "", "4;5 6"));
        check(a.length == 2 && a[1][2] == 6, "rows separated by spaces, commas and semicolons");
        rejects(() -> GramSchmidt.parseMatrix(List.of("1 2", "3")), "rows of different length");
        rejects(() -> GramSchmidt.parseMatrix(List.of("1 x")), "non-numeric entry");
        rejects(() -> GramSchmidt.parseMatrix(List.of("# only a comment")), "no rows");
    }

    private static void commandLine() throws Exception {
        var buffer = new ByteArrayOutputStream();
        var out = new PrintStream(buffer, true, StandardCharsets.UTF_8);
        check(GramSchmidt.run(new String[]{"--demo"}, out, out) == 0, "--demo succeeds");
        String text = buffer.toString(StandardCharsets.UTF_8);
        check(text.contains("Rank: 3 of 3"), "demo reports the rank");
        check(text.contains("Checks:"), "demo prints the checks");
        check(GramSchmidt.run(new String[]{"missing-file.txt"}, out, out) == 2, "missing file is an error");
        check(GramSchmidt.run(new String[]{"--help"}, out, out) == 0, "--help");
    }

    // ---- helpers ----------------------------------------------------------

    private static void valid(double[][] a, GramSchmidt.Result r, String what) {
        check(GramSchmidt.orthogonalityError(r.q()) < 1e-12, what + ": Q has orthonormal columns");
        check(GramSchmidt.reconstructionError(a, r) < 1e-9 * Math.max(1, maxAbs(a)), what + ": QR = A");
    }

    private static double maxAbs(double[][] a) {
        double m = 0;
        for (double[] row : a)
            for (double v : row)
                m = Math.max(m, Math.abs(v));
        return m;
    }

    private static boolean near(double x, double y) {
        return Math.abs(x - y) < 1e-12;
    }

    private static void rejects(Runnable action, String what) {
        try {
            action.run();
            check(false, what + " should be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, what);
        }
    }

    private static void check(boolean ok, String what) {
        checks++;
        if (!ok) {
            failures++;
            System.err.println("FAILED: " + what);
        }
    }
}
