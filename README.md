# EnerRadTool

```bash
# memo git push and branch 
cd existing_repo
git remote add origin https://git.gns.cri.nz/solenc/enerradtool.git
git branch -M master
git push -uf origin master
```

##### Phase 0 TODO list
- [x] Draft new code structure
- [x] Re-arange files
- [ ] Rename variables / functions etc
- [ ] Make documentation in functions
- [ ] New class
	- [x] parameters
	- [ ] status
	- [x] event
- [ ] Easier call
	- [x] add argv
	- [ ] make a makefile to start the env
	- [ ] make the do_smg procedure files
	- [x] Make pre-made parameters files (scenarios) + desc
- [ ] Output
	- [x] Add theta in result file
	- [ ] More contrib_save files and detailed results  files
	- [ ] Add spectra contrib
- [ ] Change querys / pre process of data process
	- [x] raw data file
	- [x] make a preprocessing function
	- [ ] make a case for synthetics

***
# README

## Description
- RCET project
- look for teleseismic data on FDSN, correct signal and filter for noise, before estimating energy flux at all station, rupture time and mean energy of the event

## Output
- results file

## Structure of the code

## Installation
###### System
Python 3.9.1
###### Requirements
see requierements file (made from winpython/spyder)
- Numpy
- Obspy
- Pygmt
- Tqdm

## Procedure
###### For a real time implementation
python main_single_event.py path2folder -rt
###### For a catalog 
-> Later

## Roadmap
#### Current version : 0.4.0
#### Planed version
- 0.4.1: Add plots
- 0.4.2: Make the loop version for catalog
#### Ideas and discussion
-

## Authors and acknowledgment

|    Name | Surname |               Email                |        Site         |
| -------:|:------- |:----------------------------------:|:-------------------:|
| CHANONY | Solen   | solen.chanony@postgrad.otago.ac.nz | Dunedin, Geol dept. |
|     FRY | Bill    |          b.fry@gns.cri.nz          |  LowerHutt, Avalon  |
| ANDREWS | Jen     |       jen.andrews@gns.cri.nz       |  LowerHutt, Avalon  |
|   ZWECK | Chris   |          c.zweck@gns.cri.nz        |  LowerHutt, Avalon  |

## License
ask Jen

