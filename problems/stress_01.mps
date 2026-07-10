NAME          STRESS_01
OBJSENSE
 MIN
ROWS
 N  OBJ
 G  C1
 G  C2
 G  C3
 L  C4
 L  C5
 L  C6
 L  C7
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ                      5
    X1        C1                       2
    X1        C2                       1
    X1        C4                       3
    X1        C6                       1
    X2        OBJ                      4
    X2        C1                       1
    X2        C2                       3
    X2        C4                       2
    X2        C5                       1
    X3        OBJ                      6
    X3        C1                       2
    X3        C3                       3
    X3        C4                       1
    X3        C7                       1
    Z1        OBJ                      8
    Z1        C1                       2
    Z1        C4                       1
    Z1        C5                       1
    Z2        OBJ                      7
    Z2        C2                       2
    Z2        C4                       1
    Z2        C6                       1
    Z3        OBJ                      9
    Z3        C3                       2
    Z3        C4                       1
    Z3        C7                       1
    MARK0001  'MARKER'                 'INTEND'
    Y1        OBJ                      1
    Y1        C1                       1
    Y1        C5                       1
    Y2        OBJ                      2
    Y2        C2                       1
    Y2        C3                       1
    Y2        C6                       1
    Y2        C7                       1
RHS
    RHS1      C1                       8
    RHS1      C2                       9
    RHS1      C3                       7
    RHS1      C4                      12
    RHS1      C5                       5
    RHS1      C6                       6
    RHS1      C7                       6
BOUNDS
 UP BND1      X1                       4
 UP BND1      X2                       5
 UP BND1      X3                       4
 BV BND1      Z1
 BV BND1      Z2
 BV BND1      Z3
 UP BND1      Y1                       4
 UP BND1      Y2                       4
ENDATA