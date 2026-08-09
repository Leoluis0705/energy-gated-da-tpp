# Deliberate exclusions

The package does not contain:

- VASP executables or source code;
- POTCAR contents or other licensed PAW datasets;
- SSH passwords, private keys, access tokens, or server credentials;
- large intermediate model checkpoints created after every acquisition round;
- raw OUTCAR, vasprun.xml, WAVECAR, or CHGCAR files;
- transient caches and Python bytecode.

The four retained relaxed structures, compact DFT audit table, reference
energies, histories, and figure source tables are sufficient to inspect the
reported evidence. Full raw VASP outputs should be deposited separately where
license and repository limits permit.

