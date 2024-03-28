import argparse
import os 

def parser_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--nPeriods', type=int, default=6, help='Number of periods for attending the emergency')
    parser.add_argument('--nProfiles', type=int, default=6, help='Number of profiles that exists in the roster')
    parser.add_argument('--maxPeriods', type=int, default=4, help='Maximum number of periods that a person can work')
    parser.add_argument('--minPeriods', type=int, default=2, help='Minimum number of periods that a person can work')
    parser.add_argument('--nPeople', type=int, default= 180,  help='Number of people that exists in the roster')
    parser.add_argument('--Discount', type=int, default=20, help='Discount for the cost of the charter')
    parser.add_argument('--nDiscount', type=int, default=10, help = ' Minimum number of persons for the discount')
    parser.add_argument('--ratio', type=int, default=30, help='Ratio of the people for each profile')

    parser.add_argument('--nCharter', type=int, default=2, help='Number of charter that exists in the roster')
    parser.add_argument('--minCapacity', type=list, default=[4,15], help='Minimum capacity of the charter')
    parser.add_argument('--maxCapacity', type=list, default=[20,90], help='Maximum capacity of the charter')
    parser.add_argument('--CostCharter1', type=int, default=750, help='Cost of the first charter')
    parser.add_argument('--CostCharter2', type=int, default=180, help='Cost of the second charter')

    parser.add_argument('--profile_path', type=str, default='profiles_EMT2.xlsx', help='Path that contains the profiles file')
    parser.add_argument('--output_path', type=str, default='Simulate_data_2.xlsx', help='Path for the output files')

    parser.add_argument('--seed', type=int, default=0, help='Seed for the random generator')

    parser.add_argument('--w1', type=int, default=1, help='Weight for the cost')
    parser.add_argument('--w2', type=int, default=1, help='Weight for infeasibility')
    parser.add_argument('--w3', type=int, default=1, help='Weight for second objective')
    parser.add_argument('--ww1', type=int, default=1, help='Weight for chartered flights')
    parser.add_argument('--ww2', type=int, default=1, help='Weight for regular flights')
    parser.add_argument('--ww3', type=int, default=1, help='Weight for number of roles')
    parser.add_argument('--ww4', type=int, default=1, help='Weight for availability')

    args = parser.parse_args()
    return args
