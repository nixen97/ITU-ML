require(Matrix)

A <- matrix(c(1, 2, 3, 4), ncol = 2, byrow = T)


R <- as.matrix(A)

n <- ncol(A)
m <- nrow(A)

H <- list()

if (m > n) {
  c <- n
} else {
  c <- m
}

for (k in 1:c) {
  
  x <- R[k:m, k]
  e <- as.matrix(c(1, rep(0, length(x) - 1)))
  vk <- sign(x[1]) * sqrt(sum(x^2)) * e + x
  
  hk <- diag(length(x)) - 2 * as.vector(vk %*% t(vk) %*% vk)
  if (k > 1) {
    hk <- bdiag(diag(k-1), hk)
  }
  
  H[[k]] <- hk
  
  R <- hk %*% R
}

Q <- Reduce("%*%", H)
res <- list('Q'=Q, 'R'=R)

print(res)
