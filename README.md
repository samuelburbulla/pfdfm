# PFDFM

This source code accompanies the paper:

_Samuel Burbulla, Luca Formaggia, Christian Rohde, and Anna Scotti: Modeling fracture propagation in poro-elastic media combining phase-field and discrete fracture models._


## Installation

To run this code, you need a working installation of [dune-mmesh](https://github.com/samuelburbulla/dune-mmesh).

If all requirements are met you can use
````
pip install dune-mmesh==1.3.2
````

## Running the examples

All examples can be executed by
````
python run.py
````

If you want to run a specific example, use
````
python poroelasticity.py [--problem <number>]
````
where <number> is 1 for `tip` or 2 for `joining` example.
