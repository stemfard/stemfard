def generate_ivp_table(t, y_approx, y_exact=None, decimal_points=8):
    data = {'Time (t)': np.round(t, decimal_points),
            'Approximated (yi)': np.round(y_approx, decimal_points)}
    
    if y_exact is not None:
        data['Exact solution(y)'] = np.round(y_exact, decimal_points)
        data['Error: |y - yi|'] = np.round(np.abs(y_exact - y_approx), decimal_points)
    
    df = pd.DataFrame(data)
    return df, df.style
