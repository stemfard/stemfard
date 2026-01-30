from functools import cached_property
from numpy import around, nan, vstack
from pandas import DataFrame

from stemfard.core.models import CoreTablesRawLatexCSV
from stemfard.stats.descriptives._base import BaseDescriptives

from stemfard.core._strings import str_remove_tzeros
from stemfard.core.convert import dframe_to_array, result_to_csv


class QtnDownloadableTables(BaseDescriptives):

    @cached_property
    def _table_qtn(self) -> CoreTablesRawLatexCSV:
        
        if self.freq is not None: # avoid `if freq`, it will crush
            arr = vstack(tup=(self._data, self.freq)).T
            table_qtn_df = DataFrame(
                data=around(arr, self.decimals),
                columns = [self.param_name, "frequency"]
            )
        else:
            table_qtn_df = DataFrame(
                data=round(self._data, self.decimals),
                columns = [self.param_name]
            )
        
        table_qtn_df_rowise = table_qtn_df.T
        
        row_names = [self.param_name]
        if self.freq is not None:
            row_names.append("Frequency")
        table_qtn_df_rowise.index = row_names
        table_qtn_df_rowise.columns = range(1, table_qtn_df_rowise.shape[1] + 1)
        
        # ComputedDataModel for `data`
        raw_data = str_remove_tzeros(table_qtn_df.values.tolist())
        latex_data = dframe_to_array(
            df=table_qtn_df,
            include_index=False,
            outer_border=True,
            inner_vlines=True
        )

        latex_data_rowise = dframe_to_array(
            df=table_qtn_df_rowise,
            include_index=True,
            outer_border=True,
            inner_vlines=True
        )
        latex_data_rowise = (
            latex_data_rowise
            .replace("r|", "c|")
            .replace(f"\\mathrm{{Frequency}}", f"\\hline\\mathrm{{Frequency}}")
            .replace(f"\\mathrm{{}} & 1", "\\qquad i & 1", 1)
        )
        csv_data = result_to_csv(obj=table_qtn_df)
        
        return CoreTablesRawLatexCSV(
            raw=raw_data,
            latex=latex_data,
            rowise=latex_data_rowise,
            csv=csv_data
        )
        
    
    @cached_property
    def _tables_downloadable(self) -> CoreTablesRawLatexCSV:
        
        assumed_mean_rnd = self.assumed_mean_rnd
        
        if self.freq is None:
            if self.assumed_mean:
                arr = vstack(tup=(self._data, self.tvalues)).T
                arr = vstack(tup=(arr, [nan, self.total_tvalues]))
                table_df = DataFrame(
                    data=around(arr, self.decimals),
                    columns = [self.param_name, self.tname]
                )
                t_csv_replaced = f", x - {assumed_mean_rnd}"
                t_latex_replaced = f"& x_{{i}} - {assumed_mean_rnd}"
                raw_calc = str_remove_tzeros(table_df.values.tolist())
                latex_calc = dframe_to_array(
                    df=table_df,
                    include_index=False,
                    outer_border=True,
                    inner_vlines=True
                )
                
                latex_calc = (
                    latex_calc
                    .replace(
                        "\\\\\n\t &",
                        f"\\\\\n\t\\hline\n\t & \\sum (x_{{i}} - {assumed_mean_rnd}) = "
                    )
                    .replace(f"& \\mathrm{{t}}", t_latex_replaced)
                )
                csv_calc = result_to_csv(obj=table_df).replace(", t", t_csv_replaced)
            else:
                return None # no `freq` and no `assumed_mean`
        else: # `freq` is not None
            if self.assumed_mean:
                arr = vstack(
                    tup=(self._data, self.tvalues, self.freq, self.fxt)    
                ).T
                arr = vstack(
                    tup=(arr, [nan, nan, self.total_freq, self.total_fxt])
                )
                table_df = DataFrame(
                    data=around(arr, self.decimals),
                    columns = [self.param_name, "t", "f", "ft"]
                )
                t_csv_replaced = f", t = x - {assumed_mean_rnd}"
                t_latex_replaced = f"& \\mathrm{{t = x - {assumed_mean_rnd}}} &"
            else:
                arr = vstack(tup=(self._data, self.freq, self.fxt)).T
                arr = vstack(
                    tup=(arr, [nan, self.freq.sum(), self.total_fxt])
                )
                table_df = DataFrame(
                    data=around(arr, self.decimals),
                    columns = [self.param_name, "f", "fx"]
                )
                t_csv_replaced = ""
                t_latex_replaced = ""
                
            # ComputedDataModel for `calculations`
            raw_calc = str_remove_tzeros(table_df.values.tolist())
            latex_calc = dframe_to_array(
                df=table_df,
                include_index=False,
                outer_border=True,
                inner_vlines=True
            )
            latex_calc = (
                latex_calc
                .replace("\\\\\n\t &", "\\\\\n\t\\hline\n\t &")
                .replace(f"& \\mathrm{{t}} &", t_latex_replaced)
                .replace("& \\mathrm{f", "& \\qquad\\mathrm{f")
            )
            
            split_str = "\\\\\n\t\\hline\n\t &"
            latex_left_right = latex_calc.split(split_str)
            if len(latex_left_right) == 2:
                latex_left, latex_right = latex_left_right
                latex_right = latex_right.split("&")
                if len(latex_right) == 2:
                    latex_right[0] = f"\\sum \\mathrm{{f}} = {latex_right[0]}"
                    latex_right[1] = f"\\sum \\mathrm{{fx}} = {latex_right[1]}"
                else:
                    latex_right[1] = f"\\sum \\mathrm{{f}} = {latex_right[1]}"
                    latex_right[2] = f"\\sum \\mathrm{{ft}} = {latex_right[2]}"
                
                latex_calc = latex_left + split_str + " & ".join(latex_right)
            
            csv_calc = result_to_csv(obj=table_df).replace(", t", t_csv_replaced)
        
        return CoreTablesRawLatexCSV(
            raw=raw_calc,
            latex=latex_calc,
            csv=csv_calc
        )