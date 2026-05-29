# Notes

## Issues with Umbrella Sampling:

- it is not guaranteed, that relevant protein motions happen during the same
  time scale as the simulation time per window!
- same problem as with metadynamics: conformational changes in the residues
  might not be represented well enough in the umbrella sampling
- it is kind of logical that if metadynamics with a set of CVs does not work,
  umbrella sampling will not work either

Started US for 1 ns from reactant, TS and product
