from pandas import Series
from numpy import array, sqrt

from stemfard.core._type_aliases import SequenceArrayLike
from verifyparams import verify_membership, verify_numeric, verify_series

from stemfard.core.utils_classes import ResultDict


LOCATION_MEASURES = ['mean', 'median', 'mode']
DISPERSION_MEASURES = ['var', 'std', 'cv', 'sem']
POSITION_MEASURES = [
    'min', 'max', 'range', 'percentiles', 'p25', 'p75', 'iqr', 'iqd'
]
DISTRIBUTION_MEASURES = ['skew', 'kurt']
OTHER_MEASURES = ['n', 'sum', 'tally']

DESCRIPTIVE_MEASURES = (
    LOCATION_MEASURES + DISPERSION_MEASURES + POSITION_MEASURES 
    + DISTRIBUTION_MEASURES + OTHER_MEASURES
)

VALID_STATISTICS = [
    'location', 'dispersion', 'position', 'distribution', 'others'
    'mean', 'median', 'mode',
    'var', 'std', 'sem', 'cv',
    'min', 'max', 'range', 'p25', 'q1', 'p75', 'q3', 'iqr',
    'skew', 'kurt'
]

   
def get_samples_stats(kwargs, is_stats: bool = False):
    """
    Validate sample1 and sample2.
    """
    valid_values = (
        ['n1', 'std1', 'n2', 'std2'] if is_stats else ['sample1', 'sample2']
    )
    verify_membership(
        value=list(kwargs),
        valid_items=valid_values,
        param_name="kwargs"
    )
    if is_stats:
        n1 = verify_numeric(
            is_positive=True,
            is_integer=True,
            value=kwargs.get('n1'),
            param_name='n1'
        )
        std1 = verify_numeric(
            is_positive=True,
            is_integer=False,
            value=kwargs.get('std1'),
            param_name='std1'
        )
        n2 = verify_numeric(
            is_positive=True,
            is_integer=True,
            value=kwargs.get('n2'),
            param_name='n2'
        )
        std2 = verify_numeric(
            is_positive=True,
            is_integer=False,
            value=kwargs.get('std2'),
            param_name='std2'
        )
    else:
        sample1 = verify_series(
            is_dropna=True,
            value=kwargs.get('sample1'),
            param_name='sample1'
        )
        sample2 = verify_series(
            is_dropna=True,
            value=kwargs.get('sample2'),
            param_name='sample2'
        )
        
        n1 = len(sample1)
        n2 = len(sample2)
        std1 = Series(sample1).std(ddof=1)
        std2 = Series(sample2).std(ddof=1)
    
    return n1, std1, n2, std2


def eda_pooled_variance_sample_stats(
    is_stats: bool = False,
    is_intermediate: bool = False,
    **kwargs
) -> float:
    """
    Calculate the pooled variance for two samples.

    The pooled variance is a weighted average of the variances from
    two independent samples, assuming that the two samples have equal
    variances.

    Parameters
    ----------
    is_stats : bool, optional (default=False)
        If `True`, the function expects sample sizes and standard
        deviations as inputs.
    **kwargs
        Keyword arguments that include sample sizes and standard
        deviations. Expected keys are 'n1', 'std1', 'n2', 'std2' when
        `is_stats` is `True` and `sample1`, `sample2` otherwise.

    Returns
    -------
    pooled_var : float
        The calculated pooled variance.

    Notes
    -----
    The pooled variance is used in situations where the assumption of
    equal variances between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab as stm
    >>> from stemlab.statistical.descriptive import eda_pooled_variance_sample_stats
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Given samples
    
    >>> dfn = eda_pooled_variance_sample_stats(is_stats=False, 
    ... sample1=female, sample2=male)
    >>> print(dfn)
    
    Given statistics
    
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = eda_pooled_variance_sample_stats(is_stats=True,
    ... n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    n1, std1, n2, std2 = get_samples_stats(is_stats=is_stats, kwargs=kwargs)
    
    steps_mathjax = []
    
    if is_stats:
        sample1_name, sample2_name = kwargs.keys()
        sample1, sample2 = kwargs.values()
        sample1 = array(sample1)
        sample2 = array(sample2)
        n1, n2 = len(sample1), len(sample2)
        std1 = sample1.std(ddof=1)
        std2 = sample2.std(ddof=1)
        
        if is_intermediate:
            steps_mathjax.append(f"{sample1_name} = {sample1}")
            steps_mathjax.append(f"{sample2_name} = {sample2}")
            steps_mathjax.append(f"\\sigma_{{1}} = To update..")
    
    pooled_var = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    
    steps_mathjax.append(
        f"\\displaystyle\\sigma^{{2}} "
        f"= \\frac{{\\left(n_{{1}} - 1\\right)\\:s_{{1}}^{{2}} "
        f"+ \\left(n_{{1}} - 1\\right)\\:s_{{2}}^{{2}}}}{{n_{{1}} + {{2}} - 2}}"
    )
    steps_mathjax.append(
        f"\\displaystyle\\quad "
        f"= \\frac{{\\left({n1} - 1\\right)\\:{std1}^{{2}} "
        f"+ \\left({n1} - 1\\right)\\:{std2}^{{2}}}}{{{n1} + {n2} - 2}}"
    )
    steps_mathjax.append(
        f"\\displaystyle\\quad "
        f"= \\frac{{{(n1 - 1) * std1 ** 2} + {(n1 - 1) * std1 ** 2}}}"
        f"{{{n1 + n2 - 2}}}"
    )
    steps_mathjax.append(
        f"\\displaystyle\\quad "
        f"= {((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)}"
    )
    
    return ResultDict(pooled_var=pooled_var, steps=steps_mathjax)


def eda_pooled_variance(
    sample1: SequenceArrayLike, sample2: SequenceArrayLike
) -> float:
    """
    Calculate the pooled variance for two samples.

    The pooled variance is a weighted average of the variances from
    two independent samples, assuming that the two samples have equal
    variances.

    Parameters
    ----------
    sample1 : SequenceArrayLike
        The first sample, which can be a list or a NumPy array of
        floats.
    sample2 : SequenceArrayLike
        The second sample, which can be a list or a NumPy array of
        floats.

    Returns
    -------
    pooled_var : float
        The calculated pooled variance.

    Notes
    -----
    The pooled variance is used when the assumption of equal variances
    between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    >>> dfn = sta.eda_pooled_variance(sample1=female, sample2=male)
    >>> print(dfn)
    """
    pooled_var = eda_pooled_variance_sample_stats(
        is_stats=False, sample1=sample1, sample2=sample2
    )
    return pooled_var


def eda_pooled_variance_stats(
    n1: int, std1: float, n2: int, std2: float
) -> float:
    """
    Calculate the pooled variance using sample statistics.

    The pooled variance is a weighted average of the variances from
    two independent samples, assuming that the two samples have equal
    variances. This function uses sample sizes and standard deviations
    to calculate the pooled variance.

    Parameters
    ----------
    n1 : int
        The sample size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The sample size of the second sample.
    std2 : float
        The standard deviation of the second sample.

    Returns
    -------
    pooled_var : float
        The calculated pooled variance.

    Notes
    -----
    The pooled variance is used when the assumption of equal variances
    between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = sta.eda_pooled_variance_stats(n1, std1, n2, std2)
    >>> print(dfn)
    """
    pooled_var = eda_pooled_variance_sample_stats(
        is_stats=True, n1=n1, std1=std1, n2=n2, std2=std2
    )
    return pooled_var


def eda_standard_error_sample_stats(
    is_stats: bool = False, is_pooled: bool = False, **kwargs
) -> float:
    """
    Calculate the standard error of the difference between two means.

    This function calculates the standard error either using pooled
    variance or separately calculated variances, depending on the
    `is_pooled` parameter.

    Parameters
    ----------
    is_stats : bool, optional
        If `True`, the function expects sample sizes and standard
        deviations as inputs. Default is False.
    is_pooled : bool, optional
        If `True`, the pooled variance is used to calculate the standard
        error. Default is False.
    **kwargs
        Keyword arguments that include sample sizes and standard
        deviations when `is_stats` is True, or sample data when 
        `is_stats` is False.

    Returns
    -------
    std_error : float
        The calculated standard error of the difference between the two
        means.

    Notes
    -----
    The standard error is a measure of how much the sample mean
    difference is expected to vary from the true population mean
    difference. The pooled variance should be used when the assumption
    of equal variances between the two samples is reasonable.

    Examples
    --------
    >>> import stemlab as stm
    >>> from stemlab.statistical.descriptive import eda_standard_error_sample_stats
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Given samples
    
    >>> dfn = eda_standard_error_sample_stats(is_stats=False, 
    ... is_pooled=False, sample1=female, sample2=male)
    >>> print(dfn)
    
    Given statistics
    
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = eda_standard_error_sample_stats(is_stats=True,
    ... is_pooled=False, n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    n1, std1, n2, std2 = get_samples_stats(is_stats=is_stats, kwargs=kwargs)
    if is_pooled:
        if is_stats:
            pooled_var = eda_pooled_variance_stats(
                n1=n1, std1=std1, n2=n2, std2=std2
            )
        else:
            pooled_var = eda_pooled_variance(
                sample1=kwargs.get('sample1'), sample2=kwargs.get('sample2')
            )
        std_error = sqrt(pooled_var * (1 / n1 + 1 / n2))
    else:
        std_error = sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
    
    return std_error


def eda_standard_error(
    sample1: SequenceArrayLike, sample2: SequenceArrayLike, is_pooled=False
) -> float:
    """
    Calculate the standard error of the difference between two means.

    This function calculates the standard error using either pooled
    variance or separately calculated variances, depending on the
    `is_pooled` parameter.

    Parameters
    ----------
    sample1 : SequenceArrayLike
        The first sample, which can be a list or a NumPy array of
        numeric values.
    sample2 : SequenceArrayLike
        The second sample, which can be a list or a NumPy array of
        numeric values.
    is_pooled : bool, optional
        If `True`, the pooled variance is used to calculate the standard
        error. Default is False.

    Returns
    -------
    std_error : float
        The calculated standard error of the difference between the
        two means.

    Notes
    -----
    The standard error is a measure of how much the sample mean
    difference is expected to vary from the true population mean
    difference. The pooled variance should be used when the
    assumption of equal variances between the two samples is
    reasonable.

    Examples
    --------
    >>> import stemlab as stm
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Given samples
    
    >>> dfn = sta.eda_standard_error(is_pooled=False,
    ... sample1=female, sample2=male)
    >>> print(dfn)
    
    Given statistics
    
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = eda_standard_error_stats(is_pooled=False,
    ... n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    std_error = eda_standard_error_sample_stats(
        is_stats=False, is_pooled=is_pooled, sample1=sample1, sample2=sample2
    )
    return std_error


def eda_standard_error_stats(
    n1: int, std1: float, n2: int, std2: float, is_pooled=False
) -> float:
    """
    Calculate the standard error of the difference between two means
    using sample statistics.

    This function calculates the standard error using either pooled
    variance or separately calculated variances, depending on the
    `is_pooled` parameter. It takes the sample sizes and standard
    deviations as inputs.

    Parameters
    ----------
    n1 : int
        The sample size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The sample size of the second sample.
    std2 : float
        The standard deviation of the second sample.
    is_pooled : bool, optional
        If `True`, the pooled variance is used to calculate the standard
        error. Default is False.

    Returns
    -------
    std_error : float
        The calculated standard error of the difference between the
        two means.

    Notes
    -----
    The standard error is a measure of how much the sample mean
    difference is expected to vary from the true population mean
    difference. The pooled variance should be used when the
    assumption of equal variances between the two samples is
    reasonable.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> dfn = sta.eda_standard_error_sample_stats(is_pooled=False,
    ... n1=n1, std1=std1, n2=n2, std2=std2)
    >>> print(dfn)
    """
    std_error = eda_standard_error_sample_stats(
        is_stats=True, is_pooled=is_pooled, n1=n1, std1=std1, n2=n2, std2=std2
    )
    return std_error


def degrees_of_freedom(
    is_stats: bool = False, is_welch: bool = True, **kwargs
) -> float:
    """
    Calculate the degrees of freedom for a two-sample t-test.

    The degrees of freedom can be calculated using either the 
    Satterthwaite or Welch approximation based on the provided sample 
    statistics.

    Parameters
    ----------
    is_stats : bool, optional (default=False)
        If `True`, the function expects sample sizes and standard 
        deviations as inputs.
    is_welch : bool, optional (default=True)
        If `True`, Welch's approximation is used; otherwise, the 
        Satterthwaite approximation is used.
    **kwargs
        Keyword arguments that include sample sizes and standard 
        deviations. Expected keys are 'n1', 'std1', 'n2', 'std2' when 
        `is_stats` is True, and `sample1` and `sample2` otherwise.

    Returns
    -------
    dfn : float
        The calculated degrees of freedom.

    Notes
    -----
    - If `is_welch` is True, Welch's approximation is used, which is 
      suitable for cases with unequal variances and sample sizes.
    - If `is_welch` is False, Satterthwaite's approximation is used, 
      which is suitable for cases where the variances are unequal but 
      sample sizes are approximately equal.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    
    Welch's degrees of freedom
    
    >>> dfn = sta.degrees_of_freedom(is_stats=False, is_welch=True,
    ... sample1=female, sample2=male)
    >>> print(dfn)
    27.207532573005
    
    Satterthwaite's degrees of freedom
    
    >>> dfn = sta.degrees_of_freedom(is_stats=False, is_welch=False,
    ... sample1=female, sample2=male)
    >>> print(dfn)
    25.28023085138802
    """
    n1, std1, n2, std2 = get_samples_stats(is_stats=is_stats, kwargs=kwargs)
    if is_welch:
        dfn = (
            (((std1 ** 2 / n1 + std2 ** 2 / n2) ** 2) /
             ((std1 ** 2 / n1) ** 2 / (n1 + 1) +
              (std2 ** 2 / n2) ** 2 / (n2 + 1))) - 2
        )
    else:
        dfn = (
            ((std1 ** 2 / n1 + std2 ** 2 / n2) ** 2) /
            (((std1 ** 2 / n1) ** 2) / (n1 - 1) +
             ((std2 ** 2 / n2) ** 2) / (n2 - 1))
        )
        
    return dfn


def df_welch(
    sample1: SequenceArrayLike,
    sample2: SequenceArrayLike
) -> float:
    """
    Calculate Welch's degrees of freedom.

    Welch's approximation is used to estimate the degrees of freedom 
    for a two-sample t-test when the variances of the two samples are 
    unequal and the sample sizes may also be different.

    Parameters
    ----------
    sample1 : SequenceArrayLike
        The first sample, which can be a list or a NumPy array of 
        floats.
    sample2 : SequenceArrayLike
        The second sample, which can be a list or a NumPy array of 
        floats.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using Welch's approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.
    
    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    >>> dfn = sta.df_welch(sample1=female, sample2=male)
    >>> print(dfn)
    27.207532573005
    """
    dfn = degrees_of_freedom(
        is_stats=False, is_welch=True, sample1=sample1, sample2=sample2
    )

    return dfn


def df_welch_stats(n1: int, std1: float, n2: int, std2: float) -> float:
    """
    Calculate Welch's degrees of freedom using sample sizes and
    standard deviations.

    Welch's approximation is used to estimate the degrees of freedom
    for a two-sample t-test when the variances of the two samples are
    unequal and the sample sizes may also be different.

    Parameters
    ----------
    n1 : int
        The size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The size of the second sample.
    std2 : float
        The standard deviation of the second sample.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using Welch's approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.
    
    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> df = sta.df_welch_stats(n1, std1, n2, std2)
    >>> print(df)
    27.207532573005
    """
    dfn = degrees_of_freedom(
        is_stats=True, is_welch=True, n1=n1, std1=std1, n2=n2, std2=std2
    )
    
    return dfn


def df_satterthwaite(
    sample1: SequenceArrayLike,
    sample2: SequenceArrayLike
) -> float:
    """
    Calculate Satterthwaite's degrees of freedom.

    The Satterthwaite approximation is used to estimate the degrees
    of freedom for a two-sample t-test when the variances of the two
    samples are unequal.

    Parameters
    ----------
    sample1 : SequenceArrayLike
        The first sample, which can be a list or a NumPy array of
        floats.
    sample2 : SequenceArrayLike
        The second sample, which can be a list or a NumPy array of
        floats.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using the Satterthwaite
        approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> df = stm.dataset_read(name='scores')
    >>> female = df.loc[df['gender'] == 'Female', 'score_after']
    >>> male = df.loc[df['gender'] == 'Male', 'score_after']
    >>> dfn = sta.df_satterthwaite(sample1=female, sample2=male)
    >>> print(dfn)
    25.28023085138802
    """
    dfn = degrees_of_freedom(
        is_stats=False, is_welch=False, sample1=sample1, sample2=sample2
    )

    return dfn


def df_satterthwaite_stats(
    n1: int, std1: float, n2: int, std2: float
) -> float:
    """
    Calculate Satterthwaite's degrees of freedom using sample sizes
    and standard deviations.

    The Satterthwaite approximation is used to estimate the degrees
    of freedom for a two-sample t-test when the variances of the two
    samples are unequal, based on their sample sizes and standard
    deviations.

    Parameters
    ----------
    n1 : int
        The size of the first sample.
    std1 : float
        The standard deviation of the first sample.
    n2 : int
        The size of the second sample.
    std2 : float
        The standard deviation of the second sample.

    Returns
    -------
    dfn : float
        The estimated degrees of freedom using the Satterthwaite
        approximation.

    Notes
    -----
    This method assumes that the two samples are independent and that
    the variances are unequal.

    Examples
    --------
    >>> import stemlab.statistical as sta
    >>> n1, std1 = (13, 14.543921102152648)
    >>> n2, std2 = (17, 13.838883838025122)
    >>> df = sta.df_satterthwaite_stats(n1, std1, n2, std2)
    >>> print(df)
    25.28023085138802
    """
    dfn = degrees_of_freedom(
        is_stats=True, is_welch=False, n1=n1, std1=std1, n2=n2, std2=std2
    )

    return dfn