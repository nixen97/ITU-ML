void cMatrix::householderDecomposition(cMatrix& Q, cMatrix& R) {
    double mag, alpha;
    cMatrix u(m, 1), v(m, 1);
    cMatrix P(m, m), I(m, m);

    Q = cMatrix(m, m);
    R = *this;

    for (int i = 0; i < n; i++) { 
        u.zero(); v.zero();
        mag = 0.0;
        for (int j = i; j < m; j++) { 
            u.A[j] = R.A[j * n + i];
            mag += u.A[j] * u.A[j];
        }
        mag = sqrt(mag);
        alpha = u.A[i] < 0 ? mag : -mag;
        mag = 0.0;
        for (int j = i; j < m; j++) {
            v.A[j] = j == i ? u.A[j] + alpha : u.A[j];
            mag += v.A[j] * v.A[j];
        }
        mag = sqrt(mag);
        if (mag < 0.0000000001) continue;
        for (int j = i; j < m; j++)
            v.A[j] /= mag;

        P = I - (v * v.transpose()) * 2.0;
        R = P * R; Q = Q * P;
}
} 