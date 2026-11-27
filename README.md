# Gram-Schmidt

Gram-Schmidt orthonormalization and QR decomposition in plain Java, with no dependencies.

Given a matrix **A**, the program turns its columns into an orthonormal basis **Q** of the
space they span and computes the upper-triangular **R** with **A = QR**. It also finds the
rank, reports columns that are linear combinations of earlier ones, and verifies its
own result.

- **Modified Gram-Schmidt with re-orthogonalization** keeps Q orthonormal to machine
  precision even for ill-conditioned matrices; the textbook classical algorithm drifts there
- **Rank-deficient input** is handled: dependent columns are detected with a relative
  tolerance and skipped, instead of normalizing rounding noise into a fake basis vector
- Works for square, tall and wide matrices; reads them from a file or standard input

## Background

For columns a₁ … aₙ, Gram-Schmidt removes from each column its components along the
basis vectors found so far, then normalizes what is left:

    vⱼ = aⱼ − Σₖ (eₖ · aⱼ) eₖ        eⱼ = vⱼ / ‖vⱼ‖

The coefficients eₖ · aⱼ and ‖vⱼ‖ form R. If ‖vⱼ‖ is (almost) zero, aⱼ adds no new
direction: it is linearly dependent on the previous columns.

## Requirements

JDK 17 or newer.

## Build and run

```sh
./build.sh                      # compile
./build.sh test                 # run the unit tests
./build.sh run --demo           # decompose a built-in example
./build.sh run matrix.txt       # decompose a matrix from a file
printf '1 2 3\n4 5 6\n7 8 9\n' | java -cp build/classes GramSchmidt
```

Input is one row per line, with numbers separated by spaces, commas or semicolons; lines
starting with `#` are ignored. When typing a matrix in by hand, finish with an empty line.

## Example

```
$ printf '1 2 3\n4 5 6\n7 8 9\n' | java -cp build/classes GramSchmidt
Input A (3 x 3)
      1.000000     2.000000     3.000000
      4.000000     5.000000     6.000000
      7.000000     8.000000     9.000000

Rank: 2 of 3 column(s)
Linearly dependent column(s), skipped: 3

Orthonormal basis Q (3 x 2)
      0.123091     0.904534
      0.492366     0.301511
      0.861640    -0.301511

R (2 x 3), A = Q R
      8.124038     9.601136    11.078234
      0.000000     0.904534     1.809068

Checks: max |Q^T Q - I| = 2.22e-16,  max |Q R - A| = 0.00e+00
```

The third column equals 2 × column 2 − column 1, so the basis has only two vectors.

## Using it as a library

```java
GramSchmidt.Result r = GramSchmidt.decompose(a);   // or decompose(a, tolerance)
double[][] q = r.q();            // m x rank, orthonormal columns
double[][] rr = r.r();           // rank x n, A = Q R
int rank = r.rank();
List<Integer> skipped = r.dependent();   // 0-based column indices
```

## Tests

`./build.sh test` checks known results, rank-deficient and zero matrices, tall and wide
shapes, a 10×10 Hilbert matrix, 50 random matrices (orthonormality and QR = A to 1e-12),
input parsing and the command line.

## License

[MIT](LICENSE)
