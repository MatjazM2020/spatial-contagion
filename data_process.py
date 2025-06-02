import pandas as pd
from pathlib import Path



def aggregate_real_estate_data(base_dir_path: str, output_path: str):
    """
    Processes yearly folders under `base_dir_path`, merges 'POSLI' and 'ZEMLJISCA' CSVs
    on 'ID_POSLA', aggregates sum of 'POGODBENA_CENA_ODSKODNINA' by 'IME_KO',
    tags each record with its YEAR, and writes the concatenated result to `output_path`.
    """
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


if __name__ == "__main__":
    aggregated_df = aggregate_real_estate_data(
        base_dir_path="kpp_lj",
        output_path="aggregated_real_estate_volume_by_year.csv"
    )
    print(aggregated_df.head())