import os
from pathlib import Path

import acts
from gnn4itk_example import runGNN4ITk

if __name__ == "__main__":
    datadir = Path("gnn_data")
    datadir.mkdir(exist_ok=True, parents=True)

    gnnModel = datadir / "gnn.pt"
    moduleMap = datadir / "modulemap"
    moduleMapDoublets = Path(str(moduleMap) + ".doublets.root")
    moduleMapTriplets = Path(str(moduleMap) + ".triplets.root")
    data = datadir / "data.root"

    def check(path, url):
        if not path.exists():
            os.system(f"wget -O {str(path)} {url}")
        else:
            print(f"Found {path}, no download necessary")

    check(gnnModel, "https://cernbox.cern.ch/s/P2WabN4aBM11iGu/download")
    check(moduleMapDoublets, "https://cernbox.cern.ch/s/zNQjg6X4J9EdBvc/download")
    check(moduleMapTriplets, "https://cernbox.cern.ch/s/RcdZNmqIPhY3iPM/download")
    # ttbar: check(data, "https://cernbox.cern.ch/s/X5iF7Hn111woAPj/download")
    check(data, "https://cernbox.cern.ch/s/22hksQOOgevF7Jj")  # single muons

    runGNN4ITk(
        inputRootDump=Path(
            "/home/bhuth/eos/data_rdo_dumps/user.avallier.mc21_14TeV.900495.PG_single_muonpm_Pt10_etaFlatnp0_43.DumpGNNITk_single_muonpm_pt10_pu0_v9.1.e8481_s4149_r14697_EXT0/user.avallier.43824508.EXT0._000093.DumpGNNITk_single_muonpm_pt10_pu0_v9.1.root"
        ),
        moduleMapPath=str(moduleMap),
        gnnModel=gnnModel,
        events=1,
        logLevel=acts.logging.DEBUG,
    )
