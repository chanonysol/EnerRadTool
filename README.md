# EnerRad General information

## Description
EnerRad is a python3 code for real-time estimation of radiated seismic energy, 
given a CMT file of the seismic source focal mechanism, using a combination of 
state-of-the-art algorithms that have been adapted for the South-West Pacific .

DOI : *Submitted to SRL special edition in July 2026*

### This file table of content
1. [Structure of the code](#structure-of-the-code)
2. [Installation](#installation)
	- [System](#system)
	- [Requirements](#requirements)
	- [Git](#git)
3. [Procedure](#procedure)
	- [For a real time implementation](#for-a-real-time-implementation)
	- [For a catalog](#for-a-catalog)
4. [Roadmap](#roadmap)
	- [Current version 2.3.0](#current-version-2.3.0)
	- [Planed version](#planed-version)
	- [Ideas and discussion](#ideas-and-discussion)
5. [Authors and acknowledgment](#authors-and-acknowledgment)
6. [License](#license)

---
## Structure of the code
More details inside each folder and coming in v2.4.3
```bash
├── EnerRadTool/
│   ├── 0.init '# parameters management, templates and details'
│   ├── 1.events_functions '# event information import and FMC-master tool'
│   ├── 2.data_functions '# query, processing, useful functions'
│   ├── 3.estimation_functions '# energy calculation and Tr estimation'
│   ├── 4.PostProcess_functions '# results storage and catalogue compilation'
│   ├── 5.plot_functions '# plot functions'
│   ├── 6.database '# updating catalogue results'
│   ├── do_query.py 
│   ├── do_preprocessing.py
│   ├── do_estimation.py
│   ├── do_correction.py
│   └── do_plot.py
├── main_EnerRad.py
├── license.txt
├── README.md
├── catalogue/
│   ├── all_contribs_corrected.dat '# station contribution per event'
│   ├── all_events_corrected.dat '# results per events'
│   ├── all_psmeca_corrected.dat '# psmeca per events'
│   └── all_xy_corrected.dat '# XY coordinates for ternary diagrams'
├── Example/ #2016 Kaikoura 4 iterations + S waves
│   ├── Data/ '# saved data used in the calculation'
│   ├── Results/ '# Result folders for different parameters'
│   ├── meca.dat 
│   ├── 201611131102A.cmt 
│   ├── param_rt_example.json 
│   ├── report_Example_folder_RT_Wprfs_D[10.80.4]_T120_new.pdf 
│   └── Chap7.pdf '# Chapter from my PhD (coming later)'
└─
```

##### Input
##### Output

---
## Installation
###### System
Python 3.9.1
###### Requirements
> *see requierements file (made from winpython/spyder)*
> - numpy
> - obspy
> - pygmt
> - tqdm

---
## Procedure

###### For a real time implementation

```bash
python main_EnerRad.py -rt "path2folder"
```

###### For a catalog
```bash
python main_EnerRad.py "path2folder" -sp -cat
```
###### For details on arguments

```bash
python main_EnerRad.py -h
```

```python
usage: main_enerrad.py [-h] [-rt] [-n] [-sp] [-cat] [-stage {1,2,3,4,5,7,42}]
                       path

positional arguments:
  path                  path to folder of event or catalog; [if rt] Folder
                        contains the cmt file and parameter file of the
                        source; [if cat] Folder contains parameters file and
                        one separate folder per source with the corresponding
                        cmt file

options:
  -h, --help            show this help message and exit
  -rt, --real_time      apply the real time code or one source
  -n, --new_param       change parameter file
  -sp, --show_param     print the code and source parameters for verification
  -cat, --catalogue     apply calculation for multiple sources in folder
  -stage {1,2,3,4,5,7,42}
                        separate stages [Default : 0 = do all] ; 1 = Only do
                        data collections ; 2 = Only do data processing ; 3 =
                        Do both data collection and processing ; 4 = Do both
                        data processing and calculation ; 5 = Do all but
                        figures ; 7 = Do calculation and plots. (If necessarry
                        previous steps have not been performed, the path and
                        missing stages will be stored in an error file) ; 42 =
                        Load event and parameters and return them for manual
                        testing
```
###### For input parameters gestion inside console / Debug mode

> `runfile` is a python command that calls the functions with arguments as an option
> ```bash
> python main_EnerRad.py [-h] [-rt] [-n] [-sp] [-cat] [-stage {1,2,3,6,42}] path
> ```
> becomes
> ```python
> runfile("main_EnerRad.py",
>       args =  "[-h] [-rt] [-n] [-sp] [-cat] [-stage {1,2,3,6,42}] path"
>       wdir='./'
> ```

If executed in console that supports the `runfile` command : 
```python
#{MODULE_DIR} = direction to module folder where main_EnerRad.py is located

runfile('{MODULE_DIR}/main_EnerRad.py', 
            args='-rt "RTdata/2026p666955/" -stage 42', 
            wdir='{MODULE_DIR}')
```

else, directly in IPython console :

```python
import sys
import os
# ---- add module to python path
CODE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
MODULE_PATH = f"{CODE_PATH}/EnerRadTool"
sys.path.insert(0, MODULE_PATH)
DIRLIST = [f"{MODULE_PATH}/{folder}"
           for folder in os.listdir(f"{MODULE_PATH}")
           if folder[:2] != "do"]
for DIR in DIRLIST:
    sys.path.insert(0, DIR)
    
parameters = Parameters(folderpath)
parameters.arglist() #show parameter file current input
parameters.param_description("name_of_the_parameter") #describe parameters and accepted values

parameters = Parameters(folderpath,new=True) #console guide to help build a new parameter file

```
---
## Roadmap
#### Current version : 2.3.2
#### Planed version
- [x] 2.0 : Added the S-wave iteration into the real-time code
	- [x] 2.0.X Minor corrections and adaptation for the RCET Kafka server
	- [x] 2.0.3 Extension of raw results output information
	- [x] 2.0.4 Correction of the processing time
- [x] 2.1 : Uncertainty evaluation
	- [x] 2.1.1 Incorporating uncertainty into raw output
	- [x] 2.1.2 Showing evolution of quality over multiple iterations
	- [x] 2.1.3 New report visual
- [x] 2.2 : Runnable code for a catalogue
	- [x] 2.2.1 Separation of the query, preprocessing, calculation and plot phases
	- [x] 2.2.2 Automatic building of catalog by stocking real-time results in the future
- [ ] 2.3 : Accessibility of the repository
    - [x] 2.3.0 Parameters rework and more user control 
	- [x] 2.3.1 Example folder with description of the process output
	- [x] 2.3.2 Catalogue results from the thesis in text-files
	
* ============================================> *Repository made public*

	- [ ] 2.3.3 Add catalogue analysis code and figures maker into the repo as callable functions
	- [ ] 2.3.4 Outline changable parameters/code etc. in code
	- [ ] 2.3.5 Adding bibliography in repository with summary from thesis
	- [ ] 2.3.6 Change plot functions to be independent 
- [ ] 2.4 : Code structure tightening and detailed documentation
    - [ ] 2.4.0 Rework plot functions call and parameters
    - [ ] 2.4.1 Cleaning remnants and stickig to recommended Python practices
    - [ ] 2.4.2 Report extra pages with explanations
    - [ ] 2.4.3 Incorporated documentation
    - [ ] 2.4.4 Implement the cleaning options to minimize data storage
- [ ] 3.0 : Improvement of the close S wave estimation
- [ ] 3.1 : Add S-wave results as a default for real-time and self-sufficient option for catalogue

#### Ideas and discussion
- **Taking request**
	- Figures
	- Description/explanation in report including plain-text
- **Intern ideas to discuss at ESNZ**
	- Public release of reports (Where? Archives? GEONET? Magnitude?)

---
## Authors and acknowledgment
The project "EnerRad" was developped during a Ph.D. thesis at the Otago University, Geology Dept., Ōtepoti - Dunedin.
It is part of the RCET and RCET-NEW programs, led by GNS and ESNZ, Aotearoa - New Zealand.


|    Name | Surname |               Email                |               Site               |
| -------:|:------- |:----------------------------------:|:--------------------------------:|
| CHANONY | Solen   | chanonysol@gmail.com		         |  France                          |
|     FRY | Bill    | bill.fry@earthsciences.nz          |  ESNZ, Avalon, LowerHutt         |
|  GORMAN | Andrew  | andrew.gorman@otago.ac.nz          |  Otago Uni, Geol dept., Ōtepoti  |
| ANDREWS | Jen     | jen.andrews@earthsciences.nz       |  ESNZ, Avalon, LowerHutt         |

## License
EnerRad is a python3 code for real-time estimation of radiated seismic energy, 
given a CMT file of the seismic source focal mechanism, using a combination of 
state-of-the-art algorithms that have been adapted for the South-West Pacific .
**Copyright (C) 2026 RCET program led by ESNZ**

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program (license.txt).  If not, see <https://www.gnu.org/licenses/>.



