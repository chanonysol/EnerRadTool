# EnerRad General information

## Description
EnerRad is a python3 code for real-time estimation of radiated seismic energy, 
given a CMT file of the seismic source focal mechanism, using a combination of 
state-of-the-art algorithms that have been adapted for the South-West Pacific .

*Under writting*
DOI : *Submitted to SRL special edition in July 2026*

### This file table of content
<!-- START doctoc -->
<!-- END doctoc -->

---
## Structure of the code
==> Coming in version 2.3
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
python main_single_event.py -rt "path2folder" -sp
```

###### For a catalog
===> Coming in 2.2

---
## Roadmap
#### Current version : 2.0.2
#### Planed version
- 2.0 : Added the S-wave iteration into the real-time code
	- 2.0.X Minor corrections and adaptation for the RCET Kafka server
	- 2.0.3 Extension of raw results output information
	- 2.0.4 Correction of the processing time
- 2.1 : Uncertainty evaluation
	- 2.1.1 Incorporating uncertainty into raw output
	- 2.1.2 Showing evolution of quality over multiple iterations
	- 2.1.3 New report visual
- 2.2 : Runnable code for a catalogue
	- 2.2.1 Separation of the query, preprocessing, calculation and plot phases
	- 2.2.2 Automatic catalogue correction, analysis and figures
	- 2.2.3 Automatic building of catalog by stocking real-time results in the future
	- 2.2.4 Implement the cleaning options to minimize data
- 2.3 : Accessibility of the repository
	- 2.3.1 Example folder with description of the process output
	- 2.3.2 Catalogue results from the thesis in text-files
	
* ============================================> Repository made public*

	- 2.3.3 Outline changable parameters/code etc. in code
	- 2.3.4 Adding bibliography in repository with summary from thesis
- 2.4 : Code structure tightening and detailed documentation
- 3.0 : Improvement of the close S wave estimation
- 3.1 : Add S-wave results as a default for real-time and self-sufficient option for catalogue
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
Copyright (C) 2026 RCET program led by ESNZ

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



