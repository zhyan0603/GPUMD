.. _kw_dump_thermo:
.. index::
   single: dump_thermo (keyword in run.in)

:attr:`dump_thermo`
===================

This keyword controls the writing of global thermodynamic quantities to the :ref:`thermo.out output file <thermo_out>`.

Syntax
------

This keyword requires the output interval (number of steps) of the global thermodynamic quantities, and optionally an element symbol::

  dump_thermo <interval> [element_symbol]

If ``element_symbol`` is provided, GPUMD will also write the summed per-atom potential energy of that element to :ref:`thermo_<element_symbol>.out <thermo_element_out>`.

Example
-------

To dump the global thermodynamic quantities every 1000 steps for a run, one can add::

  dump_thermo 1000

To additionally dump the summed potential energy of all Li atoms::

  dump_thermo 1000 Li

before the :ref:`run keyword <kw_run>`.

Caveats
-------
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.
