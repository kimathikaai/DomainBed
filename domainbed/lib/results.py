import re
import pathlib
import pandas as pd
from typing import List

VIABLE_SELECTION_METRICS = ['acc', 'f1', 'oacc', 'nacc', 'macc', 'vacc']
VIABLE_EVALUATION_METRICS = VIABLE_SELECTION_METRICS


# file scraping function that returns a row
    # Inputs: file_path
    # Returns: row as defined by ROW_TEMPLATE (perform check in function)
def scrape_latex(latex_file_path) -> List[dict]:
    
    rows = []
        
    def get_header(line):
        header = re.split(
            "\s+", 
            re.sub(r"\\textbf{(\w+)}", r"\1", line.replace("&", "").replace(r"\\", "")).strip()
            )
        dataset_list = header[1: -1]
        return header, dataset_list

    # get selection and evaluation metric
    file_name = pathlib.Path(latex_file_path).stem
    selection_metric = file_name.split('_')[-2]
    evaluation_metric = file_name.split('_')[-1]
    overlap = file_name.split('_')[-3]
    
    #print(file_name, selection_metric, evaluation_metric, overlap)
    assert selection_metric in VIABLE_SELECTION_METRICS
    assert evaluation_metric in VIABLE_EVALUATION_METRICS
    
    # read file contents
    with open(latex_file_path, 'r') as f:
        found_section = False
        found_table = False
        found_header = False
        found_table_start = False
            
        lines = f.readlines()
        for row_n, line in enumerate(lines):
            
            # print(row_n,line)
            # get to training-domain model selection section
            if (re.search("subsection{Model.*training-domain", line) or found_section):
                #if not found_section: print("*"*10, line)
                found_section = True
                if (re.search("subsubsection{Averages}", line) or found_table):
                    #if not found_table: print("*"*10, line)
                    found_table = True
                    if (re.search("textbf{Algorithm}", line) or found_header):
                        if not found_header:
                            # therefore first time to find header
                            # get header items
                            header, dataset_list = get_header(line)
                            #print(header)
                            #print(dataset_list)
                            
                            found_header = True
                            
                        # look for table beginning
                        if ('midrule' in line or found_table_start):
                            if not found_table_start:
                                found_table_start = True
                                continue
                            
                            # don't process after /bottom/rule
                            if ('bottomrule' in line):
                                # finished reading table
                                break
                            
                            # strip the row for each algorithm for
                            # value and std per dataset
                            algo_row = re.split(
                                "\s+",
                                line.replace("$\\pm$ ", ""
                                    ).replace("&", ""
                                    ).replace(r"\\", ""
                                    ).strip()
                                )
                            algorithm = algo_row.pop(0)
                            average = algo_row.pop()
                            values = algo_row
                            #print("*"*10, algo_row)
                            
                            # Support 'X'
                            org_values = values[:]
                            values = []
                            for item in org_values:
                                if item == "X":
                                    values.extend(["X", "X"])
                                else:
                                    values.append(item)

                            for idx, dataset in enumerate(dataset_list):
                                if values[idx*2] == 'X': continue
                                if float(values[idx*2]) < 0: continue
                                algorithm = "POXL" if algorithm == "XDomError" else algorithm
                                algorithm = "POXL-F" if algorithm == "XDom" else algorithm
                                algorithm = "POXL-F+B" if algorithm == "XDomBeta" else algorithm
                                algorithm = "POXL+B" if algorithm == "XDomBetaError" else algorithm
                                algorithm = "POXL-F-A" if algorithm == "SupCon" else algorithm
                                row = {
                                    'dataset': dataset, 
                                    'overlap': overlap, 
                                    'algorithm': algorithm, 
                                    'selection_metric': selection_metric, 
                                    'evaluation_metric': evaluation_metric,
                                    'selection_value': None,
                                    'evaluation_value': float(values[idx*2]),
                                    'selection_std': None,
                                    'evaluation_std': float(values[idx*2 +1]),
                                }
                                rows.append(row)
    return rows
