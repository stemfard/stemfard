from enum import Enum

from numpy import float64, nan
from numpy.typing import NDArray
from pandas import DataFrame, concat
from verifyparams import (
    verify_all_integers, verify_int_or_float, verify_len_equal,
    verify_membership
)

from stemfard.core.convert import to_numeric
from stemfard.core.results import FunctionResult


class StatisticType(Enum):
    MEAN = "mean"
    STANDARD_DEVIATION = "sd"
    PERCENTILES = "percentiles"


class AssumedMeanFormulaType(Enum):
    X_MINUS_A = "x-a"
    X_OVER_W_MINUS_A = "x/w-a"
    X_MINUS_A_OVER_W = "(x-a)/w"


class GroupedStatisticsCalculator:

    def __init__(self):
        """
        Initialize the calculator.
        """
        
    
    def compute(
        self,
        lower_limits: list[int | float] | NDArray[float64],
        upper_limits: list[int | float] | NDArray[float64],
        freq: list[int | float] | NDArray[float64],
        class_width: int | float,
        statistic: StatisticType = StatisticType.MEAN,
        decimals: int = 4,
        **kwargs
    ) -> FunctionResult:
        """
        Calculate grouped statistics with comprehensive error handling.
        
        Args:
            lower_limits: Lower class limits/boundaries
            upper_limits: Upper class limits/boundaries
            frequencies: Class frequencies
            statistic: Type of statistic to calculate
            decimals: Number of decimal places for rounding
            **kwargs: Additional parameters specific to each statistic
            
        Returns:
            FunctionResult containing answer, table, and metadata
        """
        lower_limits, upper_limits, freq, class_width, statistic = self._validate_common_params(
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            freq=freq,
            class_width=class_width,
            statistic=statistic
        )
        
        class_labels = self._generate_class_labels(
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            decimals=decimals
        )
        
        # * 0.5 for better numerical accurace than /2 (i.e. division by 2)
        midpoints = lower_limits * 0.5 + upper_limits * 0.5
        
        params_dct = {
            "class_labels": class_labels,
            "lower_limits": lower_limits,
            "upper_limits": upper_limits,
            "midpoints": midpoints,
            "freq": freq,
            "decimals": decimals
        }
        
        if statistic in ["mean", "sd"]:
            params_dct.update(
                {
                "assumed_mean": kwargs.get("assumed_mean"),
                "assumed_mean_formula": (
                    kwargs.get("assumed_mean_formula", "(x-a)/w")
                )
            }
            )
        else:
            params_dct.update(
                {
                    "percentiles": kwargs.get("percentiles"),
                }
            )
            cumfreq_curve = kwargs.get("cumfreq_curve")
            
            if cumfreq_curve:
                params_dct = {
                    "x_values": kwargs.get("x_values"),
                    "fig_width": kwargs.get("fig_width"),
                    "fig_height": kwargs.get("fig_height"),
                    "line_color": kwargs.get("line_color"),
                    "xaxis_orientation": kwargs.get("xaxis_orientation"),
                    "x_title": kwargs.get("x_title")
                }
        
        # Perform calculations based on statistic type
        
        if statistic == "mean":
            return self._calculate_mean(**params_dct)
        elif statistic == "sd":
            return self._calculate_standard_deviation(**params_dct)
        elif statistic == "percentiles":
            return self._calculate_percentiles(**params_dct)
    
    
    def _validate_common_params(
        self,
        lower_limits: list[int | float] | NDArray[float64],
        upper_limits: list[int | float] | NDArray[float64],
        freq: list[int | float] | NDArray[float64],
        class_width: int | float,
        statistic: StatisticType
    ) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        """Validate and convert input data to numpy arrays."""
        lower_limits = to_numeric(data=lower_limits, param_name="lower_limits")
        upper_limits = to_numeric(data=upper_limits, param_name="upper_limits")
        freq = to_numeric(freq, dtype=None)
        verify_all_integers(value=freq, param_name="freq")
        verify_len_equal(
            lower_limits,
            upper_limits,
            freq,
            param_names=["lower_limits", "upper_limits", "freq"]
        )
        verify_int_or_float(value=class_width, param_name="class_width")
        statistic = statistic.value
        verify_membership(
            user_input=statistic,
            valid_items=["mean", "variance", "percentiles"],
            param_name="statistic"
        )
        
        return lower_limits, upper_limits, freq, class_width, statistic
    
    
    def _generate_class_labels(
        self, 
        lower_limits: NDArray[float64], 
        upper_limits: NDArray[float64],
        decimals: int = 4
    ) -> list[str]:
        """Generate formatted class labels."""
        return [
            f"{round(lower, decimals)} - {round(upper, decimals)}"
            for lower, upper in zip(lower_limits, upper_limits)
        ]
    
    
    def _validate_assumed_mean(
        self, 
        assumed_mean: int | float, 
        assumed_mean_formula: AssumedMeanFormulaType = None
    ) -> None:
        """Validate assumed mean parameters."""
        verify_int_or_float(value=assumed_mean, param_name="assumed_mean")
        
        verify_membership(
            user_input=assumed_mean_formula,
            valid_items=["x-a", "x/w-a", "(x-a)/w"],
            param_name="assumed_mean_formula"
        )
    
    
    def _apply_assumed_mean_formula(
        self,
        midpoints: NDArray[float64],
        class_width: int | float,
        assumed_mean: int | float,
        assumed_mean_formula: AssumedMeanFormulaType,
        decimals: int = 4
    ) -> NDArray[float64]:
        """Apply assumed mean formula to midpoints."""
        if assumed_mean_formula == "x-a":
            t_values = midpoints - assumed_mean
            t_name = f"t = x - {round(assumed_mean, decimals)}"
            return t_values, t_name
        elif assumed_mean_formula == "x/w-a":
            t_values = midpoints / class_width - assumed_mean
            t_name = f"t = x / {round(class_width, decimals)} - {round(assumed_mean, decimals)}"
            return t_values, t_name
        elif assumed_mean_formula == "(x-a)/w":
            t_values = (midpoints - assumed_mean) / class_width
            t_name = f"t = (x - {round(assumed_mean, decimals)}) / {round(class_width, decimals)}"
            return t_values, t_name
    
    
    def _calculate_mean(
        self,
        class_labels: list[str],
        lower_limits: NDArray[float64],
        upper_limits: NDArray[float64],
        class_width: int | float,
        midpoints: NDArray[float64],
        freq: NDArray[float64],
        assumed_mean: int | float | None,
        assumed_mean_formula: AssumedMeanFormulaType,
        decimals: int
    ) -> FunctionResult:
        """Calculate mean for grouped data."""
        
        if assumed_mean is not None:
            if assumed_mean == -999:
                assumed_mean = 0.5 * lower_limits[0] + 0.5 * upper_limits[-1]
            else:
                assumed_mean = self._validate_assumed_mean(
                    assumed_mean=assumed_mean,
                    assumed_mean_formula=assumed_mean_formula
                )
            
            t_name, t_values = self._apply_assumed_mean_formula(
                midpoints=midpoints,
                class_width=class_width,
                assumed_mean=assumed_mean,
                assumed_mean_formula=assumed_mean_formula,
                decimals=decimals
            )

            total_freq = sum(freq)
            weighted_sum = sum(t_values * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                t_name: t_values, 
                "Frequency (f)": freq,
                "ft": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                t_name: [nan], 
                "Frequency (f)": [total_freq],
                "ft": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return FunctionResult(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": assumed_mean,
                    "formula": assumed_mean_formula,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum,
                }
            )
        else:
            total_freq = sum(freq)
            weighted_sum = sum(midpoints * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                "Frequency (f)": freq,
                "fx": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                "Frequency (f)": [total_freq],
                "fx": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return FunctionResult(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": None,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum
                }
            )
            
    
    def _calculate_standard_deviation(
        self,
        class_labels: list[str],
        lower_limits: NDArray[float64],
        upper_limits: NDArray[float64],
        class_width: int | float,
        midpoints: NDArray[float64],
        freq: NDArray[float64],
        assumed_mean: int | float | None,
        assumed_mean_formula: AssumedMeanFormulaType,
        decimals: int = 4
    ) -> FunctionResult:
        """Calculate variance for grouped data."""
        
        if assumed_mean is not None:
            if assumed_mean == -999:
                assumed_mean = 0.5 * lower_limits[0] + 0.5 * upper_limits[-1]
            else:
                assumed_mean = self._validate_assumed_mean(
                    assumed_mean=assumed_mean,
                    assumed_mean_formula=assumed_mean_formula
                )
            
            t_name, t_values = self._apply_assumed_mean_formula(
                midpoints=midpoints,
                class_width=class_width,
                assumed_mean=assumed_mean,
                assumed_mean_formula=assumed_mean_formula,
                decimals=decimals
            )

            total_freq = sum(freq)
            weighted_sum = sum(t_values * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                t_name: t_values, 
                "Frequency (f)": freq,
                "ft": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                t_name: [nan], 
                "Frequency (f)": [total_freq],
                "ft": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return FunctionResult(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": assumed_mean,
                    "formula": assumed_mean_formula,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum,
                }
            )
        else:
            total_freq = sum(freq)
            weighted_sum = sum(midpoints * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                "Frequency (f)": freq,
                "fx": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                "Frequency (f)": [total_freq],
                "fx": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return FunctionResult(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": None,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum
                }
            )
    
    
    def _calculate_percentiles(
        self,
        lower_limits: NDArray[float64],
        upper_limits: NDArray[float64],
        frequencies: NDArray[float64],
        class_labels: list[str],
        decimals: int,
        **kwargs
    ) -> FunctionResult:
        """Calculate percentiles for grouped data."""
        pass


def sta_eda_grouped_mean(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    class_width: int | float,
    assumed_mean: int | float | None,
    assumed_mean_formula: AssumedMeanFormulaType = AssumedMeanFormulaType.X_MINUS_A_OVER_W,
    decimals: int = 4,
) -> float:
    """
    Convenience function to compute grouped mean.
    
    Args:
        lc_limits: Lower class limits
        uc_limits: Upper class limits
        freq: Frequencies
        decimals: Number of decimal places
        raise_on_error: Whether to raise exceptions or return NaN
        
    Returns:
        Calculated mean
    """
    compute_class = GroupedStatisticsCalculator()
    result = compute_class.compute(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        class_width=class_width,
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )
    
    return result


def sta_eda_grouped_sd(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    class_width: int | float,
    assumed_mean: int | float | None,
    assumed_mean_formula: AssumedMeanFormulaType = AssumedMeanFormulaType.X_MINUS_A_OVER_W,
    decimals: int = 4,
) -> float:
    """
    Convenience function to compute grouped mean.
    
    Args:
        lc_limits: Lower class limits
        uc_limits: Upper class limits
        freq: Frequencies
        decimals: Number of decimal places
        raise_on_error: Whether to raise exceptions or return NaN
        
    Returns:
        Calculated mean
    """
    compute_class = GroupedStatisticsCalculator()
    result = compute_class.compute(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        class_width=class_width,
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )
    
    return result


def sta_eda_grouped_percentiles(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    class_width: int | float,
    decimals: int = 4,
) -> float:
    """
    Convenience function to compute grouped mean.
    
    Args:
        lc_limits: Lower class limits
        uc_limits: Upper class limits
        freq: Frequencies
        decimals: Number of decimal places
        raise_on_error: Whether to raise exceptions or return NaN
        
    Returns:
        Calculated mean
    """
    compute_class = GroupedStatisticsCalculator()
    result = compute_class.compute(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        class_width=class_width,
        decimals=decimals
    )
    
    return result


    