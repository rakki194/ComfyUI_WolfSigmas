#!/usr/bin/env python3
import torch

# ==================================================================================================
# USER INPUT REQUIRED HERE (NOW POPULATED FROM TABLE 3 of AYS Paper)
# ==================================================================================================
# Please populate the 'RAW_AYS_PAPER_SCHEDULES' dictionary below with the data from
# Table 2 and any other relevant tables from the Align Your Steps (AYS) paper.
#
# Each entry should be a key (e.g., "AYS_SD15_5step", "AYS_SDXL_10step_Table3")
# and the value should be another dictionary with two keys:
#   "sigmas": A list of sigma values (float) in descending order as they appear in the paper.
#             These are the *actual* sigma values, not timesteps.
#   "original_max": The sigma_max value for which this schedule was originally designed or listed
#                   in the paper (e.g., 14.615 for SD1.5, 700.669 for an SDXL example).
#                   This is important for the WolfSigmaAYSPaperSchedulePicker node to correctly
#                   scale the schedule if a different target_sigma_max is provided by the user.
#
# Example:
# RAW_AYS_PAPER_SCHEDULES = {
#     "AYS_SD15_5step_Table2_Example": {
#         "sigmas": [14.615, 3.535, 1.491, 0.742, 0.156], # Replace with actual values
#         "original_max": 14.615
#     },
#     "AYS_SDXL_10step_Table3": { # This is the one we used for AYS_CHROMA_SIGMAS_BASE (normalized)
#         # Original values before normalization to 1.0 (approximate based on previous findings)
#         "sigmas": [700.669, 302.750, 180.780, 104.560, 64.340, 41.320, 26.610, 18.220, 11.220, 5.417, 1.390],
#         "original_max": 700.669 # The first sigma in the list
#     },
#     # Add more schedules from the paper here...
#     # "AYS_SomeOtherModel_Nstep_TableX": {
#     #     "sigmas": [..., ..., ...],
#     #     "original_max": ...
#     # },
# }

RAW_AYS_PAPER_SCHEDULES = {
    "AYS_StableDiffusion_1.5_10step_Table3": {
        "sigmas": [
            14.615,
            6.475,
            3.861,
            2.697,
            1.886,
            1.396,
            0.963,
            0.652,
            0.399,
            0.152,
            0.029,
        ],
        "original_max": 14.615,
    },
    "AYS_SDXL_10step_Table3": {
        # Note: Paper lists this with original_max 14.615, which seems to be a common AYS base for SD models.
        "sigmas": [
            14.615,
            6.315,
            3.771,
            2.181,
            1.342,
            0.862,
            0.555,
            0.380,
            0.234,
            0.113,
            0.029,
        ],
        "original_max": 14.615,
    },
    "AYS_DeepFloyd_IF_Stage1_10step_Table3": {
        "sigmas": [
            160.41,
            8.081,
            3.315,
            1.885,
            1.207,
            0.785,
            0.553,
            0.293,
            0.186,
            0.030,
            0.006,
        ],
        "original_max": 160.41,
    },
    "AYS_StableVideoDiffusion_10step_Table3": {
        "sigmas": [
            700.00,
            54.5,
            15.886,
            7.977,
            4.248,
            1.789,
            0.981,
            0.403,
            0.173,
            0.034,
            0.002,
        ],
        "original_max": 700.00,
    },
    # --- USER: If you find other schedules in Table 2 or elsewhere, add them here ---
    # Example:
    # "AYS_SomeOtherModel_Nstep_TableX": {
    #     "sigmas": [..., ..., ...],
    #     "original_max": ...
    # },
}

# ==================================================================================================
# SCRIPT LOGIC (Usually no need to modify below this line)
# ==================================================================================================


def format_schedule_dict_string(schedules_dict):
    output_lines = ["AYS_PAPER_SCHEDULES = {"]
    if not schedules_dict:  # Handle empty dict case gracefully
        output_lines.append(
            "    # Populated by generate_ays_constants.py - NO SCHEDULES FOUND/DEFINED"
        )
    for name, data in schedules_dict.items():
        output_lines.append(f'    "{name}": {{')
        sigmas_str = ", ".join(map(str, data["sigmas"]))
        output_lines.append(f'        "sigmas": [{sigmas_str}],')
        output_lines.append(f'        "original_max": {data["original_max"]},')
        output_lines.append("    },")
    # Remove trailing comma from the last schedule entry if dict is not empty
    if schedules_dict and output_lines[-1].endswith(","):
        output_lines[-1] = output_lines[-1][:-1]
    output_lines.append("}")
    return "\n".join(output_lines)


if __name__ == "__main__":
    if not RAW_AYS_PAPER_SCHEDULES:
        print(
            "--------------------------------------------------------------------------"
        )
        print("ERROR: RAW_AYS_PAPER_SCHEDULES is empty in the script!")
        print("This script expects this dictionary to be populated with schedule data.")
        print("Please check the script '''%(file_name)s''' and ensure data is present.")
        print(
            "--------------------------------------------------------------------------\n"
        )
    elif (
        len(RAW_AYS_PAPER_SCHEDULES) == 1
        and "Example" in list(RAW_AYS_PAPER_SCHEDULES.keys())[0]
    ):
        print(
            "--------------------------------------------------------------------------"
        )
        print("WARNING: RAW_AYS_PAPER_SCHEDULES might still contain only example data.")
        print("Please verify the script '''%(file_name)s''' and ensure")
        print(
            "the RAW_AYS_PAPER_SCHEDULES dictionary contains actual data from the AYS paper."
        )
        print(
            "--------------------------------------------------------------------------\n"
        )

    print("# Please copy the following dictionary definition into your")
    print("# custom_nodes/ComfyUI_WolfSigmas/wolf_sigma_constants.py file,")
    print("# replacing the existing AYS_PAPER_SCHEDULES placeholder.\n")

    formatted_string = format_schedule_dict_string(RAW_AYS_PAPER_SCHEDULES)
    print(formatted_string)

    # Verification (optional, can be commented out)
    if RAW_AYS_PAPER_SCHEDULES:
        print("\n\n# --- Verification (Example of how data might be accessed) ---")
        try:
            # Create the dict from the generated string to test it
            temp_globals = {}  # Use a temporary dict for exec's globals
            exec(formatted_string, temp_globals)
            VERIFY_AYS_PAPER_SCHEDULES = temp_globals["AYS_PAPER_SCHEDULES"]

            for name in VERIFY_AYS_PAPER_SCHEDULES.keys():
                print(f'Schedule: "{name}"')
                print(
                    f'  Num sigmas (raw): {len(VERIFY_AYS_PAPER_SCHEDULES[name]["sigmas"])}'
                )
                print(
                    f'  Original max: {VERIFY_AYS_PAPER_SCHEDULES[name]["original_max"]}'
                )
                # Check if sigmas are sorted (optional, but good practice)
                s = VERIFY_AYS_PAPER_SCHEDULES[name]["sigmas"]
                is_sorted_desc = all(s[i] >= s[i + 1] for i in range(len(s) - 1))
                if not is_sorted_desc and len(s) > 1:
                    print(
                        f'  WARNING: Sigmas for "{name}" might not be in descending order!'
                    )
        except Exception as e:
            print(f"Error during verification: {e}")

    print(
        "\n# Note: The 'sigmas' list should contain N sigma values for an N-step schedule."
    )
    print("# The WolfSigmaAYSPaperSchedulePicker node will add the final '0.0' sigma.")
    print(
        "'''%(file_name)s'''" % {"file_name": __file__}
    )  # Print filename for easy editing
