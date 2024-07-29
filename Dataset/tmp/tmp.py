from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class KineticParameter:
     value: float
     substrate: str
     organism: str
     uniprot: str
     commentary: str


def extract_uniprot_accessions(
     proteins: dict[str, list[dict[str, str | list[str]]]],
) -> dict[str, set[str]]:
     res = {}

     for k, vs in proteins.items():
         _accs = set()
         for v in vs:
             if isinstance(v, list):
                 _accs.update(v)
                 continue
             if v.get("source", "") != "uniprot":
                 continue
             if (accessions := v.get("accessions")) is None:
                 continue
             _accs.update(accessions)
         if len(_accs) > 0:
             res[k] = _accs
     return res


def extract_organism(organisms: dict[str, dict[str, str]]) -> dict[str,
str]:
     return {
         k: value for k, v in organisms.items() if (value :=
v.get("value")) is not None
     }


def filter_mutant_and_recombinant(df: pd.DataFrame) -> pd.DataFrame:
     s = df["commentary"].str
     return df.loc[~s.contains("mutant") ^ s.contains("recombin")]


def _read_kinetic_parameter(
     enzyme: dict,
     group: str,
     organisms: dict[str, str],
     uniprot: dict[str, set[str]],
) -> pd.DataFrame:
     pars = []
     for v in enzyme.get(group, {}):
         if (_value := v.get("num_value")) is None:
             continue
         if (_substrate := v.get("value")) is None:
             continue
         _comment = v.get("comment", "")

         for idx in v.get("proteins", []):
             if (accession := uniprot.get(idx)) is None:
                 continue
             if (organism_name := organisms.get(idx)) is None:
                 continue

             pars.append(
                 KineticParameter(
                     value=_value,
                     substrate=_substrate,
                     organism=organism_name,
                     uniprot=next(
                         iter(accession)
                     ),  # FIXME: better way of deciding which accession
                     commentary=_comment,
                 ),
             )

     return pd.DataFrame(pars)


def read_km_and_kcat(enzyme: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
     kms = []
     organisms = extract_organism(enzyme.get("organisms", {}))
     uniprot = extract_uniprot_accessions(enzyme.get("proteins", {}))
    #  input("duuj")
     kms = _read_kinetic_parameter(
         enzyme=enzyme,
         group="km_value",
         organisms=organisms,
         uniprot=uniprot,
     )
     kcats = _read_kinetic_parameter(
         enzyme=enzyme,
         group="turnover_number",
         organisms=organisms,
         uniprot=uniprot,
     )

    #  input("sfjknjdksj")
     return kms, kcats


cache_dir = 'cache_dir'
# cache_dir.mkdir(exist_ok=True, parents=True)

# with (cache_dir / "brenda_2023_1.json").open() as fp:
    #  data1: dict[str, dict] = json.load(fp)["data"]
data1 = None
with open('/home/yashjonjale/Documents/Dataset/cache_dir/brenda_2023_1.json') as f:
    data1 = json.load(f)
    data  = data1['data']
# print(data)
# input()
for ec, enzyme in data.items():
     kms, kcats = read_km_and_kcat(enzyme)
     kms.to_json(cache_dir + f"/{ec}-km.json")
     kcats.to_json(cache_dir + f"/{ec}-kcat.json")