.. _thermo_element_out:
.. index::
   single: thermo_<element_symbol>.out (output file)

``thermo_<element_symbol>.out``
===============================

This file is written when :ref:`dump_thermo <kw_dump_thermo>` is given with an element symbol.
It contains the summed potential energy of all atoms of the selected element.

File format
-----------

There are 3 columns in this output file::

  column   1    2    3
  quantity step time U_element

* :attr:`step` is the MD step index (starting from 1)
* :attr:`time` is the simulation time (in fs)
* :attr:`U_element` is the summed potential energy (in eV) of atoms with the selected element symbol

For example, with ``dump_thermo 1000 Li``, the file name is ``thermo_Li.out``.
