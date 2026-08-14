{
  description = "Keras and PyTorch model parsers for TMVA SOFIE, formerly part of ROOT";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          let
            pkgs = nixpkgs.legacyPackages.${system};

            # TensorFlow is not available for Python 3.14, the current default
            # python3 in nixpkgs, so we pin Python 3.13.
            python = pkgs.python313;

            # ROOT has to be built against the same Python version, otherwise
            # PyROOT is not importable from our interpreter.
            root = pkgs.root.override { python3 = python; };

            sofie-extra-parsers = python.pkgs.buildPythonPackage {
              pname = "sofie-extra-parsers";
              version = "0.1.0";
              pyproject = true;
              src = self;
              build-system = [ python.pkgs.hatchling ];
              dependencies = [ python.pkgs.numpy ];
              # ROOT is a runtime dependency, but it is not a Python package;
              # it is put on PYTHONPATH by ROOT's setup hook (thisroot.sh).
              propagatedBuildInputs = [ root ];
              pythonImportsCheck = [ "sofie_extra_parsers" ];
            };

            pythonEnv = python.withPackages (ps: [
              sofie-extra-parsers
              ps.numpy
              ps.keras
              ps.tensorflow
              ps.torch
              ps.onnx
              ps.pytest
            ]);
          in
          f {
            inherit
              pkgs
              python
              root
              sofie-extra-parsers
              pythonEnv
              ;
          }
        );
    in
    {
      packages = forAllSystems (
        { sofie-extra-parsers, root, ... }:
        {
          default = sofie-extra-parsers;
          inherit root;
        }
      );

      devShells = forAllSystems (
        { pkgs, root, pythonEnv, ... }:
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              root
            ];
            # Put the in-repo sources first so that they shadow the installed
            # package during development. ROOT's setup hook already sources
            # thisroot.sh, which makes the ROOT module importable.
            shellHook = ''
              export PYTHONPATH="$PWD/python:$PYTHONPATH"
            '';
          };
        }
      );
    };
}
