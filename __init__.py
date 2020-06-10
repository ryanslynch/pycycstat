"""pycycstat
=========

Python tools for cyclostationary signal processsing (CSP)

Cyclostationary signals are those whose statistical properties are 
period or quasi-periodic.  Many human-generated signal and some 
naturally occurring processes exhibit cyclostationarity, making it
useful for signal identification and classification.

Contents
--------
   estimators  Functions for estimation cyclostationary parameters
   signal      Signal generators for CSP
   utils       Utility functions

Author
------
   Ryan S. Lynch
   Assistant Scientist
   Green Bank Observatory
   155 Observatory Road
   PO Box 2
   Green Bank, WV, 24401-0002, USA

   +1 304-456-2357
   rlynch@nrao.edu

Acknowledgements
----------------
   I am deeply grateful to Dr. Chad Spooner, author
   cyclostationary.blog, for his writing on CSP and his help in
   developing and debugging this package.  Please visit and support
   https://cyclostationary.blog for more information.

License
-------
   Copyright (C) 2019  Ryan S. Lynch

   This program is free software: you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see
   <https://www.gnu.org/licenses/>.
"""
import pkgutil

__all__ = []
for _loader, _module_name, _is_pkg in  pkgutil.walk_packages(__path__):
    __all__.append(_module_name)
    _module = _loader.find_module(_module_name).load_module(_module_name)
    globals()[_module_name] = _module
