#newton-raphson method for root-finding
def root_finder(f, XO, dX):
    X = X0
    fX = f(X, t)[1]
    tol = 1e-6
    print(fX)

    for iteration in range(100):
        if abs(fX) < tol:
            return X
    
        fpX = dX(X, t)[1]
        if fpX < tol:
            break
        
        X = X - fX/fpX
        fX = f(X)[1]
    
    return X
            
root_finder(lambda U, f: shoot(f, U), [1,1,18], dX)
        