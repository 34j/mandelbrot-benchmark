{
  description = "A flake providing a dev shell for Numba with CUDA without installing Numba via nix. Also supports PyTorch yet being minimal for Numba with CUDA.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Adjust if needed
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
      };
      cudatookit-with-cudart-to-lib64 = pkgs.symlinkJoin {
        name = "cudatoolkit";
        paths = with pkgs.cudaPackages; [
          cudatoolkit
          (pkgs.lib.getStatic cuda_cudart)
        ];
        postBuild = ''
          ln -s $out/lib $out/lib64
        '';
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          cudatoolkit
          cudaPackages.cudnn
        ];
        shellHook = ''
          # Required for both PyTorch and Numba to find CUDA
          export CUDA_PATH=${cudatookit-with-cudart-to-lib64}

          # Set CC to GCC 13 to avoid the version mismatch error
          export CC=${pkgs.gcc13}/bin/gcc
          export CXX=${pkgs.gcc13}/bin/g++
          export PATH=${pkgs.gcc13}/bin:$PATH

          # Required for both PyTorch and Numba, adds necessary paths for dynamic linking
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              "/run/opengl-driver" # Needed to find libGL.so, required by both PyTorch and Numba
            ]
          }:$LD_LIBRARY_PATH

          # Set LIBRARY_PATH to help the linker find the CUDA static libraries
          export LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              pkgs.cudatoolkit
            ]
          }:$LIBRARY_PATH
        '';
      };
    };
}
