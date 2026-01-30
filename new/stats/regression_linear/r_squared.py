class LinearRegression:
    ...
    # Existing properties like sst, ssr, mse, etc.
    
    @property
    def r_squared(self) -> float:
        """Coefficient of determination"""
        return self.ssr / self.sst

    @property
    def r_squared_rnd(self) -> float:
        return float(around(self.r_squared, self.params.decimals))

    @property
    def adj_r_squared(self) -> float:
        """Adjusted R-squared for multiple regression"""
        return 1 - (1 - self.r_squared) * (self.n - 1) / (self.n - self.k)

    @property
    def adj_r_squared_rnd(self) -> float:
        return float(around(self.adj_r_squared, self.params.decimals))
