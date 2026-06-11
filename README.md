# EnerRadTool

```bash
# memo git push and branch
cd existing_repo
git remote add origin https://git.gns.cri.nz/solenc/enerradtool.git
git branch -M master
git push -uf origin master
```

##### Phase 2 TODO list
- [ ] Clear functions unused
- [ ] Add catalogue plot functions
- [ ] 


***
# README

## Description
- RCET programm
- look for teleseismic data on FDSN, correct signal and filter for noise, before estimating energy flux at all station, rupture time and mean energy of the event
DOI : (see later)

## Output
- results file

## Structure of the code

## Installation
###### System
Python 3.9.1
###### Requirements
see requierements file (made from winpython/spyder)
- numpy
- obspy
- pygmt
- tqdm

## Procedure
###### For a real time implementation
python main_single_event.py path2folder -rt
###### For a catalog
-> Later

## Roadmap
#### Current version : 2.0.0
#### Planed version
- 2.0.1: Add clean option to minimize data
- 2.0.2: Add options for catalogue
- 2.1.0: Improvement for S waves
#### Ideas and discussion
-

## Authors and acknowledgment

|    Name | Surname |               Email                |        Site         |
| -------:|:------- |:----------------------------------:|:-------------------:|
| CHANONY | Solen   | chanonysol@gmail.com		           | Dunedin, Geol dept. |
|     FRY | Bill    | b.fry@gns.cri.nz 			             |  LowerHutt, Avalon  |
| ANDREWS | Jen     |                                    |  LowerHutt, Avalon  |
|         | Chris   |                                    |  LowerHutt, Avalon  |

## License
ask Jen

