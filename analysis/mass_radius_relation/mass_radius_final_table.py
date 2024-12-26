"""
mass_radius_final_table.py
Combines all the separately generated mass-size fit tables into one
"""
import sys

output = sys.argv[1]
legus_full = sys.argv[2]
legus_young = sys.argv[3]
legus_agesplit = sys.argv[4]
legus_mw = sys.argv[5]
legus_m31 = sys.argv[6]
legus_mw_m31 = sys.argv[7]

# start output file, then make the header
fit_out_file = open(output, "w")
fit_out_file.write("\t\\begin{tabular}{llcccc}\n")
fit_out_file.write("\t\t\\toprule\n")
fit_out_file.write(
    "\t\tSelection & "
    "$N$ & "
    "$\\beta$: Slope & "
    "$R_4$: $\\reff$(pc) at $10^4\Msun$ & "
    "Intrinsic Scatter & "
    "$\log{M}$ percentiles: 1--99 \\\\ \n"
)
fit_out_file.write("\t\t\midrule\n")

# helper functions
def copy_single_file(in_file_loc):
    """
    Copies the table held in one file into the output table

    :param in_file_loc: Location of one file to copy over
    :type in_file_loc: str
    :return: None
    """
    with open(in_file_loc, "r") as in_file:
        for line in in_file:
            if line.strip() != "":
                fit_out_file.write(line)


def out_file_spacer():
    fit_out_file.write("\t\t\midrule\n")


# then go through everything and copy them over
copy_single_file(legus_full)
copy_single_file(legus_young)
out_file_spacer()
copy_single_file(legus_agesplit)
out_file_spacer()
copy_single_file(legus_mw)
copy_single_file(legus_m31)
copy_single_file(legus_mw_m31)

# then finalize the output file
fit_out_file.write("\t\t\\bottomrule\n")
fit_out_file.write("\t\end{tabular}\n")
fit_out_file.close()
