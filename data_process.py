import pandas as pd
from pathlib import Path
import pandas as pd
import geopandas as gpd


def aggregate_real_estate_data(base_dir_path: str, output_path: str):
    base_dir = Path(base_dir_path)
    annual_aggregations = []
    for year_folder in sorted(base_dir.iterdir()):
        if not year_folder.is_dir(): 
            continue
        
        year = int(year_folder.name.split('_')[-1][:4])

        posli_file = next(year_folder.glob('*POSLI*.csv'))
        zemljisca_file = next(year_folder.glob('*ZEMLJISCA*.csv'))

        df_posli = pd.read_csv(posli_file)
        df_zemljisca = pd.read_csv(zemljisca_file)

        merged = pd.merge(df_zemljisca, df_posli, on='ID_POSLA', how='inner')

        agg = (
            merged
            .groupby('IME_KO', as_index=False)['POGODBENA_CENA_ODSKODNINA']
            .sum()
            .rename(columns={'POGODBENA_CENA_ODSKODNINA': 'TOTAL_VOLUME'})
        )
        agg['YEAR'] = year
        annual_aggregations.append(agg)

    result_df = pd.concat(annual_aggregations, ignore_index=True)

    result_df.to_csv(output_path, index=False)
    print(f"Aggregated data saved to: {output_path}")
    return result_df




def calculate_neighbours():
    path_delistavb = (
        "/Users/matjazmadon/Development/spatial-contagion/"
        "kpp_lj/kpp_lj_2024/ETN_061_2024_KPP_2024_DELISTAVB_20250524.csv"
    )
    
    path_shapefile = (
        "/Users/matjazmadon/Development/spatial-contagion/"
        "shp_lj/KN_SLO_KAT_OBCINE_KATASTRSKE_OBCINE_poligon.shp"
    )

    df_delist = pd.read_csv(path_delistavb, dtype={"SIFRA_KO": str})
    if "OBCINA" not in df_delist.columns or "SIFRA_KO" not in df_delist.columns:
        raise KeyError("Expected columns 'OBCINA' and 'SIFRA_KO' in the DELISTAVB CSV.")

    df_lj_delist = df_delist[df_delist["OBCINA"].str.strip().str.lower() == "ljubljana"].copy()
    lj_sifra_ko = df_lj_delist["SIFRA_KO"].unique().tolist()

    gdf_all = gpd.read_file(path_shapefile)

    gdf_all = gdf_all.rename(columns={"NAZIV": "IME_KO", "SIFKO": "SIFRA_KO"})
    gdf_all["SIFRA_KO"] = gdf_all["SIFRA_KO"].astype(str)

    gdf_lj = gdf_all[gdf_all["SIFRA_KO"].isin(lj_sifra_ko)].copy().reset_index(drop=True)
    if gdf_lj.empty:
        raise ValueError("No matching cadastral‐unit codes found in the shapefile for Ljubljana.")

    gdf_lj["geometry"] = gdf_lj["geometry"].buffer(0)

    sindex = gdf_lj.sindex

    neighbors = {}

    for idx, row in gdf_lj.iterrows():
        this_code = row["SIFRA_KO"]
        this_geom = row["geometry"]
        possible_idxs = list(sindex.intersection(this_geom.bounds))
        possible = gdf_lj.iloc[possible_idxs]
        possible = possible[possible.index != idx]

        touches_mask = possible.geometry.touches(this_geom)
        touching = possible.loc[touches_mask, ["SIFRA_KO", "IME_KO"]]

        neighbors[this_code] = touching["SIFRA_KO"].tolist()

    records = []
    for idx, row in gdf_lj.iterrows():
        code_a = row["SIFRA_KO"]
        name_a = row["IME_KO"]
        for code_b in neighbors[code_a]:
            if code_a < code_b:
                name_b = gdf_lj.loc[gdf_lj["SIFRA_KO"] == code_b, "IME_KO"].values[0]
                records.append({
                    "SIFRA_KO_A": code_a,
                    "IME_KO_A": name_a,
                    "SIFRA_KO_B": code_b,
                    "IME_KO_B": name_b,
                })

    df_neighbors = pd.DataFrame.from_records(records,
                                             columns=[
                                               "SIFRA_KO_A", "IME_KO_A",
                                               "SIFRA_KO_B", "IME_KO_B"
                                             ])

    output_path = "ljubljana_cadastral_neighbors.csv"
    df_neighbors.to_csv(output_path, index=False)
    print(f"Finished. Neighbor list saved to {output_path}")
    print(f"Found {len(df_neighbors)} unique adjacency pairs among Ljubljana cadastral units.")



if __name__ == "__main__":
    aggregated_df = aggregate_real_estate_data(
        base_dir_path="kpp_lj",
        output_path="aggregated_real_estate_volume_by_year.csv"
    )
    calculate_neighbours()
    print(aggregated_df.head())