from numpy import around

from stemfard.core._html import html_bg_level2
from stemfard.core.constants import StemConstants
from stemfard.stats.descriptives._base import BaseDescriptives
from stemfard.stats.descriptives._tables import QtnDownloadableTables


FORMULAS_MEAN: dict[str, str] = {
    "data_and_no_freq": (
        f"x_{{1}}, \\: x_{{2}}, \\: x_{{3}}, \\: \\cdots, \\: x_{{n}}"
    ),
    "data_and_freq": (
        f"\\begin{{array}}{{|l|c|c|c|c|c|}} \\hline "
        f"\\mathrm{{Data}} & x_{{1}} & x_{{2}} & x_{{3}} & \\cdots & x_{{n}} \\\\ \\hline "
        f"\\mathrm{{Frequency}} & f_{{1}} & f_{{2}} & f_{{3}} & \\cdots & f_{{n}} \\\\ \\hline "
        f"\\end{{array}}"
    ),
    "mean_and_no_assumed_mean_and_no_freq": (
        f"\\displaystyle\\bar{{x}} = \\frac{{ \\sum x_{{i}}}}{{n}}"
    ),
    "mean_and_no_assumed_mean_and_freq": (
        f"\\displaystyle\\bar{{x}} "
        f"= \\frac{{ \\sum\\mathrm{{fx}}}}{{ \\sum\\mathrm{{f}}}}"
    ),
    "mean_and_assumed_mean_and_no_freq": (
        f"\\displaystyle\\bar{{x}} = A + \\frac{{ \\sum (x_{{i}} - A)}}{{n}} "
    ),
    "mean_and_assumed_mean_and_freq": (
        f"\\displaystyle\\bar{{x}} "
        f"= A + \\frac{{ \\sum\\mathrm{{ft}}}}{{ \\sum\\mathrm{{f}}}}"
    )
}


class ArithmeticMeanSteps(BaseDescriptives):
    
    def bg_arithmetic_mean(self) -> list[str]:
        
        steps_mathjax: list[str] = []

        if self.freq is None:
            steps_mathjax.append(
                "Consider the following set of data with \\( n \\) values."
            )
            mformula = FORMULAS_MEAN["data_and_no_freq"]
            steps_mathjax.append(f"\\[ {mformula} \\]")
        else:
            steps_mathjax.append(
                "Consider the following set of \\( n \\) values and their "
                "respective frequencies."
            )
            mformula = FORMULAS_MEAN["data_and_freq"]
            steps_mathjax.append(f"\\[ {mformula} \\]")
        
        if self.freq is None:
            if self.assumed_mean:
                steps_mathjax.append(
                    f"The arithmetic mean, denoted by \\( \\bar{{x}} \\), "
                    "given an assumed mean \\( A \\), is calculated using "
                    "the following formula."
                )
                mformula = FORMULAS_MEAN["mean_and_assumed_mean_and_no_freq"]
                steps_mathjax.append(f"\\[ {mformula} \\]")
                steps_mathjax.append(
                    "where \\( A \\) is the assumed mean, \\( f \\) is the "
                    "set of frequencies and \\( t = x - A \\)."
                )
            else:
                steps_mathjax.append(
                    f"The arithmetic mean, denoted by \\( \\bar{{x}} \\), is "
                    "calculated using the following formula."
                )
                
                mformula = FORMULAS_MEAN["mean_and_no_assumed_mean_and_no_freq"]
                steps_mathjax.append(f"\\[ {mformula} \\]")
                steps_mathjax.append(
                    "where \\( x \\) and \\( f \\) are the given set of "
                    "values and frequencies respectively."
                )
        else:
            if self.assumed_mean:
                steps_mathjax.append(
                    f"The arithmetic mean, denoted by \\( \\bar{{x}} \\), "
                    "given an assumed mean \\( A \\), is calculated using "
                    "the following formula."
                )
                mformula = FORMULAS_MEAN["mean_and_assumed_mean_and_freq"]
                steps_mathjax.append(f"\\[ {mformula} \\]")
                steps_mathjax.append(
                    "where \\( A \\) is the assumed mean and \\( n \\) is "
                    "the number of values."
                )
            else:
                steps_mathjax.append(
                    f"The arithmetic mean, denoted by \\( \\bar{{x}} \\), is "
                    f"found by multiplying \\( f_{{i}} \\) by the "
                    f"corresponding \\( x_{{i}} \\), evaluating the sum of "
                    "all these products, then dividing it by the total "
                    "number of elements, \\( n \\). That is,"
                )
                mformula = FORMULAS_MEAN["mean_and_no_assumed_mean_and_freq"]
                steps_mathjax.append(f"\\[ {mformula} \\]")
        
        steps_mathjax.append(StemConstants.BORDER_HTML_BG)
        
        return steps_mathjax
    
    
    def arithmetic_mean_reusable(
        self, border_before: bool = True
    ) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        if border_before:
            steps_mathjax.append(StemConstants.BORDER_HTML)
        
        mformula = FORMULAS_MEAN["mean_and_no_assumed_mean_and_no_freq"]
        steps_mathjax.append(f"\\( {mformula} \\)")
        steps_mathjax.append(
            "\\( \\displaystyle\\quad "
            f"= \\frac{{{self.total_rnd}}}{{{self.n}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.mean_rnd} \\)")
        
        return steps_mathjax
    
    
    def mean_step_1_to_2(
        self, assumed_mean: int | float | None, mformula: str 
    ) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        temp_step = html_bg_level2(
            title="STEP 1: Count the number of elements, \\( n \\)"
        )
        steps_mathjax.append(temp_step)
        steps_mathjax.append(
            f"There are a total of \\( {self.n} \\) elements. That is, "
        )
        steps_mathjax.append(f"\\( n = {self.n} \\)")
        
        temp_step = html_bg_level2(
            title="STEP 2: Write down the arithmetic mean formula"
        )
        steps_mathjax.append(temp_step)
        
        if assumed_mean:
            steps_mathjax.append(
                "The formula for calculating the arithmetic mean of "
                "\\( n \\) values using an assumed mean \\( A \\) is as "
                "given below."
            )
            steps_mathjax.append(f"\\( \\quad {mformula} \\)")
            steps_mathjax.append(
                f"When \\( A = {self.assumed_mean_rnd} \\) and "
                f"\\( n = {self.n} \\), the above formula becomes,"
            )
            mformula = (
                mformula
                .replace("A", str(self.assumed_mean_rnd))
                .replace("n", str(self.n))
            )
            steps_mathjax.append(f"\\( \\quad {mformula} \\)")
        else:
            steps_mathjax.append(
                "The formula for calculating the arithmetic mean of "
                "\\( n \\) values is as given below."
            ) 
            steps_mathjax.append(f"\\( {mformula} \\)")
        
        return steps_mathjax

    
    def mean_and_no_assumed_mean_and_no_freq(self) -> list[str]:
        """
        Arithmetic mean only (i.e. no assumed mean, no frequencies).
        """
        steps_mathjax: list[str] = []
        mformula = FORMULAS_MEAN["mean_and_no_assumed_mean_and_no_freq"]
        
        step_temp = self.mean_step_1_to_2(
            assumed_mean=None,
            mformula=mformula
        )
        steps_mathjax.extend(step_temp)
        
        temp_step = html_bg_level2(title="STEP 3: Sum all the values")
        steps_mathjax.append(temp_step)
        steps_mathjax.append(
            f"The sum of the \\( {self.n} \\) values is calculated as "
            "follows."
        )
        
        data_sum_joined = " + ".join(map(str, self.data_rnd))
        steps_mathjax.append(f"\\( \\sum x_{{i}} = {data_sum_joined} \\)")
        steps_mathjax.append(f"\\( \\quad\\quad = {self.total_rnd} \\)")
        
        temp_step = html_bg_level2(
            title="STEP 4: Calculate the arithmetic mean"
        )
        steps_mathjax.append(temp_step)
        
        steps_mathjax.append(
            f"Evaluate the arithmetic mean by dividing the sum calculated in "
            f"\\( \\text{{STEP 3}} \\) by the number of elements found in "
            f"\\( \\text{{STEP 1}} \\). This is illustrated below."
        )
        steps_mathjax.append(f"\\( {mformula} \\)")
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= \\frac{{{self.total_rnd}}}{{{self.n}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.mean_rnd} \\)")
        
        return steps_mathjax
    
    
    def mean_and_no_assumed_mean_and_freq(self) -> list[str]:
        """
        Given data (x) and frequencies (f).
        """
        steps_mathjax: list[str] = []
        params = {
            "statistic": self.statistic,
            "data": self._data,
            "freq": self.freq,
            "assumed_mean": self.assumed_mean,
            "steps_compute": self.steps_compute,
            "steps_detailed": self.steps_detailed,
            "show_bg": self.show_bg,
            "param_name": self.param_name,
            "decimals": self.decimals
        }
        
        table_latex = QtnDownloadableTables(**params)._tables_downloadable.latex
        
        step_temp = html_bg_level2(
            title="STEP 1: Write down the arithmetic mean formula"
        )
        steps_mathjax.append(step_temp)
        steps_mathjax.append(
            "The formula for calculating the arithmetic mean is as given "
            "below."
        )
        mformula = FORMULAS_MEAN["mean_and_no_assumed_mean_and_freq"]
        steps_mathjax.append(f"\\( \\quad {mformula} \\)")
        steps_mathjax.append(
            "where \\( x \\) and \\( f \\) are the given set of values and "
            "frequencies respectively."
        )
        
        step_temp = html_bg_level2(
            title="STEP 2: Create a table for calculations"
        )
        steps_mathjax.append(step_temp)
        steps_mathjax.append(
            "The table below is generated for ease of calculations."
        )
        steps_mathjax.append(f"\\[ {table_latex} \\]")
        
        step_temp = html_bg_level2(
            title="STEP 3: Calculate the mean of \\( x \\)"
        )
        steps_mathjax.append(step_temp)
        steps_mathjax.append(
            "The mean of \\( x \\) is found by dividing the sum in column "
            f"\\( \\mathrm{{fx}} \\) by the sum in column "
            f"\\( \\mathrm{{f}} \\) as follows."
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\bar{{x}} "
            f"= \\frac{{\\sum\\mathrm{{fx}}}}{{\\sum\\mathrm{{f}}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= \\frac{{{self.total_fxt_rnd}}}{{{self.total_freq}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.mean_rnd} \\)")
        
        return steps_mathjax
    
    
    def mean_and_assumed_mean_and_no_freq(self) -> list[str]:
        """
        Arithmetic mean given data and assumed mean.
        """
        steps_mathjax: list[str] = []
        params = {
            "statistic": self.statistic,
            "data": self._data,
            "freq": self.freq,
            "assumed_mean": self.assumed_mean,
            "steps_compute": self.steps_compute,
            "steps_detailed": self.steps_detailed,
            "show_bg": self.show_bg,
            "param_name": self.param_name,
            "decimals": self.decimals
        }
        
        assumed_mean_rnd = self.assumed_mean_rnd
        table_latex = QtnDownloadableTables(**params)._tables_downloadable.latex
        
        step_temp = self.mean_step_1_to_2(
            assumed_mean=self.assumed_mean,
            mformula=FORMULAS_MEAN["mean_and_assumed_mean_and_no_freq"]
        )
        steps_mathjax.extend(step_temp)

        temp_step = html_bg_level2(
            title=f"STEP 3: Create a table for calculations"
        )
        steps_mathjax.append(temp_step)
        
        if self.is_calculate_amean:
            steps_mathjax.append(
                "Begin by approximating the assumed mean \\( A \\) from the "
                "given data using the calculation below."
            )
            
            data_min = around(self.min, self.decimals)
            data_max = around(self.max, self.decimals)
            
            steps_mathjax.append(
                "\\( \\displaystyle\\quad A "
                f"= \\frac{{\\max(x) + \\min(x)}}{{2}} "
                f"= \\frac{{{data_max} + {data_min}}}{{2}} "
                f"= {assumed_mean_rnd} \\)"
            )
        
        steps_mathjax.append(
            "The differences "
            f"\\( \\left(x_{{i}} - {assumed_mean_rnd}\\right) \\) are "
            "calculated and presented in the second column of the table "
            "below."
        )
        steps_mathjax.append(f"\\[ {table_latex} \\]")
        
        temp_step = html_bg_level2(
            title="STEP 4: Calculate the arithmetic mean"
        )
        steps_mathjax.append(temp_step)
        steps_mathjax.append(
            "The mean of \\( x \\) is then calculated by diving the sum in "
            "the second column of the table above by \\( n \\), then adding "
            "this result to the assumed mean. This is shown below."
        )
        formula_updated_with_values = (
            FORMULAS_MEAN["mean_and_assumed_mean_and_no_freq"]
                .replace("A", str(assumed_mean_rnd))
                .replace("n", str(self.n))
        )
        steps_mathjax.append(f"\\( {formula_updated_with_values} \\)")
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad = {assumed_mean_rnd} "
                f"+ \\frac{{{self.total_tvalues_rnd}}}{{{self.n}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= {assumed_mean_rnd} + {self.mean_tvalues_rnd} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.mean_rnd} \\)")
            
        return steps_mathjax
    
    
    def mean_and_assumed_mean_and_freq(self) -> list[str]:
        """
        Given data (x), frequencies (f) and assumed mean (A).
        """
        steps_mathjax: list[str] = []
        params = {
            "statistic": self.statistic,
            "data": self._data,
            "freq": self.freq,
            "assumed_mean": self.assumed_mean,
            "steps_compute": self.steps_compute,
            "steps_detailed": self.steps_detailed,
            "show_bg": self.show_bg,
            "param_name": self.param_name,
            "decimals": self.decimals
        }
        table_latex = QtnDownloadableTables(**params)._tables_downloadable.latex
        assumed_mean_rnd = self.assumed_mean_rnd
        
        step_temp = html_bg_level2(
            title="STEP 1: Write down the arithmetic mean formula"
        )
        steps_mathjax.append(step_temp)
        steps_mathjax.append(
            "The formula for calculating the arithmetic mean using an "
            "assumed mean \\( A \\) is given below."
        )
        mformula = FORMULAS_MEAN["mean_and_assumed_mean_and_freq"]
        steps_mathjax.append(f"\\( \\quad {mformula} \\)")
        
        if self.is_calculate_amean:
            steps_mathjax.append(
                "where \\( A \\) is the assumed mean, \\( f \\) is the set "
                "of frequencies and \\( t = x - A \\)."
            )
        else:
            steps_mathjax.append(
                "where \\( A \\) is the assumed mean, \\( f \\) is the set "
                f"of frequencies and \\( t = x - {assumed_mean_rnd} \\)."
            )
        
        step_temp = html_bg_level2(
            title="STEP 2: Create a table for calculations"
        )
        steps_mathjax.append(step_temp)
        
        if self.is_calculate_amean:
            steps_mathjax.append(
                "Begin by approximating the assumed mean \\( A \\) from the "
                "given data using the calculation below."
            )
            
            data_min = around(self.max, self.decimals)
            data_max = around(self.min, self.decimals)
            
            steps_mathjax.append(
                f"\\( \\displaystyle\\quad A "
                f"= \\frac{{\\max(x) + \\min(x)}}{{2}} "
                f"= \\frac{{{data_max} + {data_min}}}{{2}} "
                f"= {assumed_mean_rnd} \\)"
            )
        
        steps_mathjax.append(
            "The differences "
            f"\\( \\left(x_{{i}} - {assumed_mean_rnd}\\right) \\) are "
            "calculated and presented in the second column of the table "
            "below."
        )
        steps_mathjax.append(f"\\[ {table_latex} \\]")
        
        step_temp = html_bg_level2(
            title="STEP 3: Calculate the mean of \\( t \\)"
        )
        steps_mathjax.append(step_temp)
        steps_mathjax.append(
            "The mean of \\( t \\) is found by dividing the sum in column "
            f"\\( \\mathrm{{ft}} \\) by the sum in column "
            f"\\( \\mathrm{{f}} \\) as follows."
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\bar{{t}} "
            f"= \\frac{{\\sum\\mathrm{{ft}}}}{{\\sum\\mathrm{{f}}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= \\frac{{{self.total_fxt_rnd}}}{{{self.total_freq}}} \\)"
        )
        
        steps_mathjax.append(f"\\( \\quad = {self.mean_fxt_rnd} \\)")
        
        step_temp = html_bg_level2(
            title="STEP 4: Calculate the mean of \\( x \\)"
        )
        steps_mathjax.append(step_temp)
        
        steps_mathjax.append(
            "The mean of \\( x \\) is then found by adding the mean of "
            f"\\( t \\) calculated in \\( \\textbf{{STEP 3}} \\) to the "
            "assumed mean \\( A \\) as shown below."
        )
        steps_mathjax.append(f"\\( \\bar{{x}} = A + \\bar{{t}} \\)")
        steps_mathjax.append(
            f"\\( \\quad = {assumed_mean_rnd} + {self.mean_fxt_rnd} \\)"
        )
        # DO NOT attempt to do `self.mean_x_rnd`
        steps_mathjax.append(
            f"\\( \\quad = {self.assumed_mean_rnd + self.mean_fxt_rnd} \\)"
        )
        
        return steps_mathjax