.. _kw_dump_temp:
.. index::
   single: dump_temp (keyword in run.in)

:attr:`dump_temp`
=================

Write the system temperature and all subgroup temperatures for one grouping method to :ref:`temp.out <temp_out>`.

Syntax
------

.. code::

   dump_temp <interval> group <grouping_method>

The :attr:`interval` parameter is the output interval (number of steps).
The :attr:`grouping_method` parameter selects which grouping method is used when outputting subgroup temperatures.

The output of each sampled frame contains:

* MD step index
* global system temperature
* one temperature for each group in the selected grouping method

Example
-------

To dump every 100 steps and use grouping method 0::

  dump_temp 100 group 0

before the :ref:`run keyword <kw_run>`.

Caveats
-------
This keyword is not propagating.
That means, its effect will not be passed from one run to the next.

