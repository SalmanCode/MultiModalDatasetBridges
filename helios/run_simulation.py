import os


def run_helios_simulation(survey_path, helios_exe="helios"):
    """Run HELIOS simulation with the given survey file.
    
    Args:
        survey_path: Path to the survey XML file
        helios_exe: Path or command to HELIOS executable
    """
    cmd = f"{helios_exe} {survey_path}"
    print(f"Running: {cmd}")
    os.system(cmd)
