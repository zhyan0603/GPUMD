.. _temp_out:
.. index::
   single: temp.out (output file)

``temp.out``
============

This file contains the sampled global temperature together with subgroup temperatures for one grouping method.
It is written by :ref:`dump_temp <kw_dump_temp>`.

File format
-----------

Each row corresponds to one sampled frame.

Column layout:

* column 1: :attr:`step` (MD step index, starting from 1)
* column 2: :attr:`T_total` (global system temperature in K)
* columns 3 to :math:`(2 + N_g)`: :attr:`T_group(i)` (temperature in K of group :math:`i` in the selected grouping method), where :math:`N_g` is the number of groups in that grouping method.

In short, each line is:

.. code::

   step T_total T_group(0) T_group(1) ... T_group(Ng-1)

