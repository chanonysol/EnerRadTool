# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:01:57 2024

@author: solen
"""
import os
import re
import json
import inspect
import shutil

from copy import deepcopy

from numpy import argwhere, array, ndarray

from colorama import init as colorama_init
from colorama import Fore, Back, Style


colorama_init(autoreset=True)
PARAM_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
SCENAR_PATH = f"{PARAM_PATH}/scenarios"
def color_option(text):
    """
    Highlight the option in the description for the parameters

    Parameters
    ----------
    text : str

    """

    if text.find("\n")==-1 :
        if text.find(" ")!=-1:
            details = text.split()
            col = Fore.GREEN

            option = details[0]
            end = " ".join(details[1:])

            text = (Style.BRIGHT + Fore.BLACK + "    - "
                    + col + f"{option} " + Fore.BLACK + end)

        return text

    if text[0]!='-':
        lign = text.split('\n')
        text = lign[0]
        details = text.split()
        col = Fore.GREEN

        option = details[0]
        end = " ".join(details[1:])

        lign[0] = (Style.BRIGHT + Fore.BLACK + "     -"
                   + col + f" {option} " + Fore.BLACK + end)
        lign[1] = Style.BRIGHT + Fore.BLACK + lign[1]
        lign[-1] = lign[-1] + Style.RESET_ALL
        for i in range(1,len(lign)):
            lign[i] = "     " + lign[i].replace("-","o ")
        return '\n'.join(lign)

    ligns=[]
    for newline in text.split('\n'):
        details = newline.split()
        col = Fore.GREEN

        if details[1] == "or" and details[2] == "type":
            start = " ".join(details[:3])
            option = details[3]
            end = " ".join(details[4:])

        elif details[1] == "or":
            start = " ".join(details[:2])
            option = details[2]
            end = " ".join(details[3:])

        else :
            start = details[0]
            option = details[1]
            end = " ".join(details[2:])

        if option in ["true","false","null"]:
            col = Fore.MAGENTA

        ligns.append(Style.BRIGHT + "    "+ Fore.BLACK + start
                     + col + f" {option} "
                     + Fore.BLACK + end)
    return '\n'.join(ligns)

def find_param_file(path):
    """
    Look for existing parameter files in the folder

    Parameters
    ----------
    path : str
        Current directory or path

    Returns
    -------
    STR or ARRAY or None
        Existing file(s) or None if none exist

    """

    pattern = array([re.compile(r"^param").match(i) for i in os.listdir(path)])
    ind =  argwhere(pattern)
    if ind.size==0 :
        return None
    if ind.size>1 :
        return array(os.listdir(path))[ind[:,0]]
    return os.listdir(path)[ind[0,0]]

class Parameters:
    """
    =====
    Parameters
    =====

    Definition: Parameters(path, new=False)

    Load parameters for the session. If it doesn't exist, it will enter a user
    friendly code to create the file.

    Parameters
    ----------
    path : str
        working directory
    new : bool, optional
        Option to create a new file or use existing one.
        The default is False.

    Returns
    -------
    out : parameters
        Parameters object with all options to perform

    Notes
    -----
    Some scenarios exists in a folder that you can copy and use in your
    session. Just change the name to fit the format 'param_*.json'

    """

    def __init__(self,path,new = False, show = False):

        if '__original__' not in vars(self):
            self.read_param(path,new)

        for key,value in self.__original__.items():
            if isinstance(value,dict):
                setattr(self,key, decorators[key](path,self.__original__[key]))
                continue
            setattr(self,key, value)

        if show :
            its_fine = "n"
            while its_fine == "n":
                self.arglist()
                its_fine = input(f"{Back.GREEN + Style.DIM}Are you fine with the current parameters {Style.RESET_ALL}"
                                 +f" ({Fore.GREEN + Style.BRIGHT}y{Style.RESET_ALL})"
                                 +f" or "
                                 +f"({Fore.GREEN + Style.BRIGHT}n{Style.RESET_ALL})?"
                                 +f"{Style.RESET_ALL} -> ")
                if its_fine == "n":
                    self.read_param(path,True)


    def read_param(self,path,new):
        """
        Read the param file into a dictionnary to be exploit

        """

        param_file = self.search_file(path,new)
        with open(param_file, encoding="utf-8") as file : param = json.load(file)
        self.__original__ = param
    def search_file(self,path,new):
        """
        Search for the param file, if none create a new one, if multiple
        ask which to keep and remove all others

        """

        param_file = find_param_file(path)
        if new and param_file:
            print(f"{Back.RED}One or more param file already exist and will be deleted{Style.RESET_ALL}")
            keep_going = input(f'continue? ({Fore.GREEN}y or n{Style.RESET_ALL}) -> ')
            if keep_going == 'n' :
                raise BaseException("Process interrupted")
            if not isinstance(param_file, (list,ndarray)) : param_file = [param_file]
            for file in param_file[::-1]:
                os.remove(f"{path}/{file}")
            new_file = input("Creating a new one "
                             +f"({Fore.GREEN + Style.BRIGHT}y{Style.RESET_ALL})"
                             +f" or select a model "
                             +f"({Fore.GREEN + Style.BRIGHT}n{Style.RESET_ALL})?"
                             +f"{Style.RESET_ALL} ")
            if new_file == 'n':
                copy_file = input(f"{Back.CYAN}Choose the model to copy for this event : {Style.RESET_ALL}\n   {Fore.CYAN}- "\
                                  +f"{Fore.CYAN}\n   - ".join([scenar.replace(".json","")
                                             for scenar
                                             in os.listdir(SCENAR_PATH)]
                                            )
                                  +f"{Style.RESET_ALL}\n-> "
                                  )
                save_file = input(f"{Back.CYAN}which name should be given to the parameters file? {Style.RESET_ALL}(no .json needed)\n-> ")
                shutil.copy(f"{SCENAR_PATH}/{copy_file}.json",f"{path}/param_{save_file}.json")
            if new_file == 'y':
                save_file = input(f"{Back.CYAN}which name should be given to the parameters file? {Style.RESET_ALL}(no .json needed)\n-> ")
                self.init_param_file(path,save_file)
            param_file = "param_"+save_file+".json"

        if isinstance(param_file,ndarray) and not new :
            print(f"{Back.RED}Multiple param file already exist, you need to choose one, others are to be deleted{Style.RESET_ALL}")
            for file in param_file:
                print(Style.BRIGHT + Fore.BLACK + "  - " + file)
            file_2_keep = input(f'{Back.CYAN}which file to keep?{Style.RESET_ALL} ')
            for file in param_file[::-1]:
                if file != file_2_keep:
                    os.remove(path+file)
            print('\n')
            param_file = file_2_keep

        if not param_file:
            new_file = input("Parameters file not found - create a new one "
                             +f"({Fore.GREEN + Style.BRIGHT}y{Style.RESET_ALL})"
                             +f" or select a model "
                             +f"({Fore.GREEN + Style.BRIGHT}n{Style.RESET_ALL})?"
                             +f"{Style.RESET_ALL} ")
            if new_file == 'n':
                copy_file = input(f"{Back.CYAN}Choose the model to copy for this event : {Style.RESET_ALL}\n   {Fore.CYAN}- "\
                                  +f"{Fore.CYAN}\n   - ".join([scenar.replace(".json","")
                                             for scenar
                                             in os.listdir(SCENAR_PATH)]
                                            )
                                  +f"{Style.RESET_ALL}\n-> "
                                  )
                save_file = input(f"{Back.CYAN}which name should be given to the parameters file? {Style.RESET_ALL}(no .json needed)\n-> ")
                shutil.copy(f"{SCENAR_PATH}/{copy_file}.json",f"{path}/param_{save_file}.json")
            if new_file == 'y':
                save_file = input(f"{Back.CYAN}which name should be given to the parameters file? {Style.RESET_ALL}(no .json needed)\n-> ")
                self.init_param_file(path,save_file)
            param_file = "param_"+save_file+".json"

        return path+param_file

    def init_param_file(self,path,save_file):
        """
        User help programm to create parameter file

        Parameters
        ----------
        path : str
            working directory
        save_file : str
            name for the parameter file (without .json extension)

        """

        with open(f'{PARAM_PATH}/param_description.json', encoding="utf-8") as file : data = json.load(file)
        with open(f'{PARAM_PATH}/param_default.json', encoding="utf-8") as file : param_default = json.load(file)
        print(f"{Fore.RED+Style.BRIGHT} Entering parameters files creator {Style.NORMAL} :\n"
              +f"either type one of the given option or skip to set default values (highlighted) {Style.RESET_ALL}")
        param=deepcopy(param_default)
        for k in data.keys():
            if isinstance(data[k], dict):
                for key in data[k].keys():
                    if isinstance(data[k][key], dict) and param[k]["level"]!='personalized':continue
                    if isinstance(param[k][key], dict) :
                        print(f"{Fore.YELLOW+Style.BRIGHT}{k}{Style.RESET_ALL} :"
                              +f" {Fore.GREEN}{key} ")
                        print(f"{Style.BRIGHT+Fore.BLACK}for each type "
                              +f"{Fore.MAGENTA}true"
                              +f"{Fore.BLACK} or enter"
                              +f"{Back.WHITE}(default = false){Style.RESET_ALL}\n")

                        for i in param[k][key].keys():
                            print(Style.BRIGHT + Fore.BLACK + "   - " + i + " : " + data[k][key][i])
                            param[k][key][i] = input("your option? ")
                            if param[k][key][i] in ['true','True','t','T'] :
                                param[k][key][i]=True
                            if param[k][key][i] in ['','false','False','f','F'] : param[k][key][i]=False
                        continue
                    print(f"{Fore.YELLOW+Style.BRIGHT}{k}{Style.RESET_ALL} :"
                          +f" {Fore.GREEN}{key} "
                          +f"{Style.BRIGHT+Fore.BLACK+Back.WHITE}"
                          +f"default = {param_default[k][key]}{Style.RESET_ALL}")
                    print(Style.BRIGHT + Fore.BLACK + data[k][key])
                    param[k][key] = input("your option? ")
                    try : float(param[k][key])
                    except ValueError:
                        if param[k][key] == '' : param[k][key]=param_default[k][key]
                        if param[k][key] in ['null','none','None','n', None] : param[k][key]=None
                        if param[k][key] in ['true','True','t','T'] : param[k][key]=True
                        if param[k][key] in ['false','False','f','F'] : param[k][key]=False
                        print('\n')
                    else :
                        param[k][key]=float(param[k][key])
                        print('\n')
                continue
            print(f"{Fore.YELLOW+Style.BRIGHT}{k}{Style.RESET_ALL} "
                  +f"{Style.BRIGHT+Fore.BLACK+Back.WHITE}"
                  +f"default = {param_default[k]}{Style.RESET_ALL}")
            print(Style.BRIGHT + Fore.BLACK + data[k])
            if k == 'channel_code':
                param[k] = []
                val = input("your option (enter after each code)? ")
                while val != "":
                    param[k].append(val)
                    val = input("your option (enter after each code)? ")
                if len(param[k]) == 0: param[k] = param_default[k]
                print('\n')
                continue
            else:
                param[k] = input("your option? ")
            try : float(param[k])
            except ValueError:
                if param[k] == '' : param[k]=param_default[k]
                if param[k] in ['null','none','None','n', None] : param[k]=None
                if param[k] in ['true','True','t','T'] : param[k]=True
                if param[k] in ['false','False','f','F'] : param[k]=False
                print('\n')
            else :
                param[k]=float(param[k])
                print('\n')
            print("----------------------------------")

        # Serializing json
        json_object = json.dumps(param, ensure_ascii=False,indent=4)

        # Writing to sample.json
        with open(f"{path}/param_{save_file}.json", "w", encoding="utf-8") as outfile:
            outfile.write(json_object)

    def describe_param(self,request):
        """
        Print the description of requested parameter

        Parameters
        ----------
        request : str
            requested parameter

        """
        with open(f'{PARAM_PATH}/param_description.json', encoding="utf-8") as file : data = json.load(file)
        if request not in data:
            for key in data:
                if isinstance(data[key],dict) and request in data[key] and request!="details":
                    print(Style.DIM + Back.RED + request
                            + Style.RESET_ALL + " : "
                            + Style.DIM + Back.MAGENTA + key)
                    print(color_option(data[key][request]))
            if request in data["plot"]["details"] or request=="details":
                print(Style.DIM + Back.RED + "plot"
                        + Style.RESET_ALL + " : "
                        + Style.DIM + Back.MAGENTA + "details")
                for subkey in data["plot"]["details"]:
                    print(Style.BRIGHT + Fore.BLACK + "    -"
                    + Style.DIM + Fore.CYAN + f" {subkey} : "
                    + Style.BRIGHT + Fore.BLACK + data["plot"]["details"][subkey])
        elif isinstance(data[request],dict) :
            for key in data[request]:
                print(Style.DIM + Back.RED + request
                        + Style.RESET_ALL + " : "
                        + Style.DIM + Back.MAGENTA + key)
                print(color_option(data[request][key]))
        else:
            print(Style.DIM + Back.RED + request + Style.RESET_ALL)
            print(color_option(data[request]))
    def arglist(self, col = Style.DIM + Fore.RED,space=0):
        """
        Print all the parameters and their value

        """
        args = [i for i in inspect.getmembers(self) if not i[0].startswith('_')
                and not inspect.ismethod(i[1])]
        if space==0: print(Style.DIM + Back.MAGENTA + "ARGUMENTS LIST")
        for arg in args :
            if arg[0] in decorators:
                print(f"{' '*space}" + col + f"{arg[0]}")
                decorators[arg[0]].arglist(arg[1])
                continue
            print(f"{' '*space}" + col + f"{arg[0]}" + Style.RESET_ALL
                  + f" : {arg[1]}")

class ParamGeneral(Parameters):
    def __init__(self,path,dic):
        self.__original__ = dic
        super().__init__(path)
    def arglist(self,col = Fore.MAGENTA, space = 2):
        super().arglist(col,space)

class ParamPlot(Parameters):
    def __init__(self,path,dic):
        self.__original__ = dic
        super().__init__(path)
    def arglist(self,col = Fore.MAGENTA, space = 2):
        super().arglist(col,space)

class ParamPlotDetails(ParamPlot):
    def __init__(self,path,dic):
        self.__original__ = dic
        super().__init__(path,dic)
    def arglist(self,col = Fore.CYAN, space = 5):
        super().arglist(col,space)

class ParamDistanceMax(Parameters):
    def __init__(self,path,dic):
        self.__original__ = dic
        super().__init__(path)
    def arglist(self,col = Fore.MAGENTA, space = 2):
        super().arglist(col,space)

class ParamDeviationCorrection(Parameters):
    def __init__(self,path,dic):
        self.__original__ = dic
        super().__init__(path)
    def arglist(self,col = Fore.MAGENTA, space = 2):
        super().arglist(col,space)

class ParamRedoingResults(Parameters):
    def __init__(self,path,dic):
        self.__original__ = dic
        super().__init__(path)
    def arglist(self,col = Fore.MAGENTA, space = 2):
        super().arglist(col,space)

decorators = {}
for name,Class in zip(["general","plot","details","distance_max",
                       "deviation_correction","redoing_results"],
                      [ParamGeneral,ParamPlot,ParamPlotDetails,
                       ParamDistanceMax,ParamDeviationCorrection,
                       ParamRedoingResults]):
    decorators[name] = Class
