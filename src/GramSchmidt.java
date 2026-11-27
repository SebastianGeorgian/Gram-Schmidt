import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Gram-Schmidt orthonormalization and QR decomposition of a real matrix.
 *
 * <p>The columns of A are turned into an orthonormal basis Q of their span, together with an
 * upper-triangular R such that A = Q * R. The implementation uses <em>modified</em> Gram-Schmidt
 * with one re-orthogonalization pass ("twice is enough"), which keeps Q orthogonal to machine
 * precision even for ill-conditioned matrices, where the textbook classical algorithm drifts.
 *
 * <p>Columns that are linear combinations of the previous ones (within a relative tolerance)
 * are detected and skipped, so Q has exactly rank(A) columns.
 */
public final class GramSchmidt {

    /** Default tolerance: a column is dependent if less than this fraction of its norm is new. */
    public static final double DEFAULT_TOLERANCE = 1e-10;

    /**
     * Result of the decomposition.
     *
     * @param q              m x rank matrix with orthonormal columns
     * @param r              rank x n upper-trapezoidal matrix, A = Q * R
     * @param dependent      0-based indices of the columns of A that were linearly dependent
     */
    public record Result(double[][] q, double[][] r, List<Integer> dependent) {
        public int rank() {
            return q.length == 0 ? 0 : q[0].length;
        }
    }

    private GramSchmidt() {
    }

    public static Result decompose(double[][] a) {
        return decompose(a, DEFAULT_TOLERANCE);
    }

    public static Result decompose(double[][] a, double tolerance) {
        int m = a.length;
        if (m == 0)
            throw new IllegalArgumentException("The matrix has no rows.");
        int n = a[0].length;
        for (double[] row : a) {
            if (row.length != n)
                throw new IllegalArgumentException("All rows must have the same number of entries.");
            for (double x : row)
                if (!Double.isFinite(x))
                    throw new IllegalArgumentException("The matrix contains a non-finite value.");
        }

        List<double[]> basis = new ArrayList<>();      // orthonormal vectors found so far
        List<double[]> coefficients = new ArrayList<>(); // row k of R, one entry per column of A
        List<Integer> dependent = new ArrayList<>();

        for (int j = 0; j < n; j++) {
            double[] v = column(a, j);
            double originalNorm = norm(v);
            double[] projections = new double[basis.size()];

            for (int pass = 0; pass < 2; pass++) {
                for (int k = 0; k < basis.size(); k++) {
                    double[] e = basis.get(k);
                    double c = dot(e, v);
                    projections[k] += c;
                    for (int i = 0; i < m; i++)
                        v[i] -= c * e[i];
                }
            }
            for (int k = 0; k < basis.size(); k++)
                coefficients.get(k)[j] = projections[k];

            double remaining = norm(v);
            if (originalNorm == 0.0 || remaining <= tolerance * originalNorm) {
                dependent.add(j);
                continue;
            }
            for (int i = 0; i < m; i++)
                v[i] /= remaining;
            basis.add(v);
            double[] rRow = new double[n];
            rRow[j] = remaining;
            coefficients.add(rRow);
        }

        int rank = basis.size();
        double[][] q = new double[m][rank];
        for (int k = 0; k < rank; k++)
            for (int i = 0; i < m; i++)
                q[i][k] = basis.get(k)[i];
        return new Result(q, coefficients.toArray(new double[0][]), List.copyOf(dependent));
    }

    /** Orthonormal basis of the column space of A (the Q of the decomposition). */
    public static double[][] orthonormalize(double[][] a) {
        return decompose(a).q();
    }

    // ---- small linear-algebra helpers ------------------------------------------------

    static double[] column(double[][] a, int j) {
        double[] c = new double[a.length];
        for (int i = 0; i < a.length; i++)
            c[i] = a[i][j];
        return c;
    }

    static double dot(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++)
            s += x[i] * y[i];
        return s;
    }

    static double norm(double[] x) {
        double scale = 0;
        for (double v : x)
            scale = Math.max(scale, Math.abs(v));
        if (scale == 0)
            return 0;
        double s = 0;
        for (double v : x)
            s += (v / scale) * (v / scale); // scaled to avoid overflow and underflow
        return scale * Math.sqrt(s);
    }

    static double[][] multiply(double[][] x, double[][] y) {
        int m = x.length, inner = y.length, n = inner == 0 ? 0 : y[0].length;
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int k = 0; k < inner; k++)
                for (int j = 0; j < n; j++)
                    out[i][j] += x[i][k] * y[k][j];
        return out;
    }

    static double[][] transpose(double[][] x) {
        int m = x.length, n = m == 0 ? 0 : x[0].length;
        double[][] t = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                t[j][i] = x[i][j];
        return t;
    }

    /** Largest absolute entry of X - Y. */
    static double maxDifference(double[][] x, double[][] y) {
        double d = 0;
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[i].length; j++)
                d = Math.max(d, Math.abs(x[i][j] - y[i][j]));
        return d;
    }

    static double[][] identity(int n) {
        double[][] id = new double[n][n];
        for (int i = 0; i < n; i++)
            id[i][i] = 1;
        return id;
    }

    /** ‖QᵀQ − I‖ (max entry): how far the computed basis is from orthonormal. */
    public static double orthogonalityError(double[][] q) {
        int rank = q.length == 0 ? 0 : q[0].length;
        return maxDifference(multiply(transpose(q), q), identity(rank));
    }

    /** ‖QR − A‖ (max entry): how well the decomposition reproduces the input. */
    public static double reconstructionError(double[][] a, Result result) {
        if (result.rank() == 0)
            return maxDifference(a, new double[a.length][a[0].length]);
        return maxDifference(multiply(result.q(), result.r()), a);
    }

    // ---- command line ---------------------------------------------------------------

    static final String USAGE = """
            Usage: java GramSchmidt [FILE | --demo | --help]

            Reads a matrix (one row per line, numbers separated by spaces, commas or
            semicolons) from FILE, or from standard input when no FILE is given, and
            prints the orthonormal basis Q, the triangular factor R with A = QR, the
            rank, and checks of the result. Lines starting with # are ignored.

            Example:  printf '1 1 0\\n1 0 1\\n0 1 1\\n' | java GramSchmidt
            """;

    static final double[][] DEMO = {
            {1, 1, 0},
            {1, 0, 1},
            {0, 1, 1},
    };

    public static void main(String[] args) throws IOException {
        System.exit(run(args, System.out, System.err));
    }

    static int run(String[] args, PrintStream out, PrintStream err) throws IOException {
        double[][] a;
        try {
            if (args.length > 1)
                throw new IllegalArgumentException("Expected at most one argument.");
            if (args.length == 1 && (args[0].equals("-h") || args[0].equals("--help"))) {
                out.print(USAGE);
                return 0;
            }
            if (args.length == 1 && args[0].equals("--demo")) {
                a = DEMO;
            } else if (args.length == 1) {
                a = parseMatrix(Files.readAllLines(Path.of(args[0]), StandardCharsets.UTF_8));
            } else {
                if (System.console() != null)
                    out.println("Enter the matrix one row per line, then an empty line (or Ctrl+D):");
                a = parseMatrix(readUntilBlank(new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))));
            }
        } catch (IllegalArgumentException | IOException e) {
            err.println("GramSchmidt: " + e.getMessage());
            err.println("Run with --help for usage.");
            return 2;
        }

        Result result;
        try {
            result = decompose(a);
        } catch (IllegalArgumentException e) {
            err.println("GramSchmidt: " + e.getMessage());
            return 2;
        }
        printReport(out, a, result);
        return 0;
    }

    static void printReport(PrintStream out, double[][] a, Result result) {
        int n = a[0].length;
        out.printf("Input A (%d x %d)%n", a.length, n);
        printMatrix(out, a);
        out.printf("%nRank: %d of %d column(s)%n", result.rank(), n);
        if (!result.dependent().isEmpty()) {
            var cols = result.dependent().stream().map(j -> String.valueOf(j + 1)).toList();
            out.println("Linearly dependent column(s), skipped: " + String.join(", ", cols));
        }
        if (result.rank() == 0) {
            out.println("\nEvery column is zero: there is no basis to compute.");
            return;
        }
        out.printf("%nOrthonormal basis Q (%d x %d)%n", a.length, result.rank());
        printMatrix(out, result.q());
        out.printf("%nR (%d x %d), A = Q R%n", result.rank(), n);
        printMatrix(out, result.r());
        out.printf("%nChecks: max |Q^T Q - I| = %.2e,  max |Q R - A| = %.2e%n",
                orthogonalityError(result.q()), reconstructionError(a, result));
    }

    static void printMatrix(PrintStream out, double[][] x) {
        for (double[] row : x) {
            var sb = new StringBuilder(" ");
            for (double v : row) {
                double shown = Math.abs(v) < 5e-13 ? 0.0 : v; // hide -0.000000 and 1e-17 noise
                sb.append(String.format(Locale.ROOT, " %12.6f", shown));
            }
            out.println(sb);
        }
    }

    static List<String> readUntilBlank(BufferedReader in) throws IOException {
        List<String> lines = new ArrayList<>();
        for (String line; (line = in.readLine()) != null; ) {
            if (line.isBlank() && !lines.isEmpty())
                break;
            lines.add(line);
        }
        return lines;
    }

    static double[][] parseMatrix(List<String> lines) {
        List<double[]> rows = new ArrayList<>();
        for (String raw : lines) {
            String line = raw.strip();
            if (line.isEmpty() || line.startsWith("#"))
                continue;
            String[] parts = line.split("[\\s,;]+");
            double[] row = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                try {
                    row[i] = Double.parseDouble(parts[i]);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Not a number on row " + (rows.size() + 1) + ": '" + parts[i] + "'");
                }
            }
            if (!rows.isEmpty() && row.length != rows.get(0).length)
                throw new IllegalArgumentException("Row " + (rows.size() + 1) + " has " + row.length
                        + " entries, but row 1 has " + rows.get(0).length + ".");
            rows.add(row);
        }
        if (rows.isEmpty())
            throw new IllegalArgumentException("No matrix rows were given.");
        return rows.toArray(new double[0][]);
    }
}
