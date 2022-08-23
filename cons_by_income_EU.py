

main_path = './'
import sys
sys.path.append(main_path+"lib/")

# load helper functions
from cons_funs_EU import*

# load consumption by income decentile data (Eurostat)
cons_eu = pd.read_csv('data/consumption_EU_icio.csv')
groups = cons_eu['group'].unique()

# load sector names
sector_names = pd.read_csv('consumption_analysis/sector_labels_agg.txt', sep='\t')

# countries for which we have consumption by decentile:
sample = ['AUT', 'BGR', 'CYP', 'DNK', 'ESP', 'EST', 'FIN', 'GRC', 'HUN', 'IRL',
          'LVA', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN']

# get solutions from model
sectors, countries, pq_isj_C, pq_isj_B, w_j_C, w_j_B, iot_isjk_C, iot_isjk_B, \
emint_is_B, carb_cost_isj_C, p_tilde_is, P_tilde_sj, sigma_s, so, ba = \
    get_model_solutions(results_path = main_path+'results/',
                        data_path = main_path+'data/',
                        dir_num = 1,
                        year =2015,
                        carb_cost_list = [1e-4],
                        eta_path = ['uniform_elasticities_4.csv'],
                        sigma_path = ['uniform_elasticities_4.csv'],
                        taxed_countries_list = [None],
                        taxing_countries_list = [None],
                        taxed_sectors_list = [None],
                        specific_taxing_list = [None],
                        fair_tax_list = [False],
                        years = [2015],
                        sample_countries = sample)

dims = [sectors, countries, groups, sample, sector_names]


# disaggregate Eurostat consumption data by origin
pq_isnj_B, pq_snj_eu = disaggregate_data(dims, pq_isj_B, cons_eu)

# compute income change
I_tilde_nj, I_nj_B, I_nj_C, tax_rev_isj = \
    get_income_change(dims, pq_snj_eu, w_j_B, w_j_C, iot_isjk_C, pq_isj_C, emint_is_B, carb_cost_isj_C)

# solve consumer problem
t_isj, q_tilde_isnj, pq_isnj_C, T_snj_C, tau_snj, tax_nj, U_tilde_snj, U_tilde_nj = \
    get_consumer_problem_solution(dims, p_tilde_is, emint_is_B, carb_cost_isj_C, sigma_s, P_tilde_sj, I_tilde_nj, pq_isnj_B, pq_snj_eu, I_nj_C)






# -- - - - - - - -- -  -- - - - - -
s, k, i, j, n, sector_group_sample_df = getdims(dims)

test = sector_group_sample_df.copy()
test['consumption'] = np.sum(pq_isnj_C, axis=0).reshape(s*n*j)

test1 = test.groupby(['industry','group'], as_index=False).sum()
fig = plt.figure(figsize=(16,8))
sns.barplot(x='industry',y='consumption', hue='group', data=test1)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show

test2 = test.groupby(['industry','country'], as_index=False).sum()
fig = plt.figure(figsize=(16,8))
sns.barplot(x='industry',y='consumption', hue='country', data=test2)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show



utilitytest = sector_group_sample_df.copy().filter(['group','country']).drop_duplicates()
utilitytest['utility'] = (U_tilde_nj-1).reshape(n*j)
fig = plt.figure(figsize=(16,8))
sns.barplot(x='country',y='utility', hue='group', data=utilitytest)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show

incometest = sector_group_sample_df.copy().filter(['group','country']).drop_duplicates()
incometest['incomechange'] = I_tilde_nj.reshape(n*j)
fig = plt.figure(figsize=(16,8))
sns.barplot(x='country',y='incomechange', hue='group', data=incometest)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show


taxtest = sector_group_sample_df.copy().filter(['group','country']).drop_duplicates()
taxtest['tax'] = tax_nj.reshape(n*j)
fig = plt.figure(figsize=(16,8))
sns.barplot(x='country',y='tax', hue='group', data=taxtest)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show