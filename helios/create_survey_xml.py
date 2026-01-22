def create_survey_xml(bridge, positions, output_path):
    """Create survey XML file for a bridge.
    
    Args:
        bridge: Dictionary containing bridge parameters (bridge_id)
        positions: Dictionary with scanner positions for all legs
        output_path: Path where the survey XML file will be saved
    """
    bridge_id = bridge['bridge_id']
    
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
	<!-- Default scanner settings: -->
    <scannerSettings id="profile1" active="true" pulseFreq_hz="100000" scanFreq_hz="120" scanAngle_deg="100" headRotatePerSec_deg="10.0"/>
    <survey name="TLS_{bridge_id}" scene="./data/scenes/TLS_{bridge_id}_scene.xml#TLS_{bridge_id}" platform="data/platforms.xml#tripod" scanner="data/scanners_tls.xml#riegl_vz400">
        <FWFSettings binSize_ns="0.2" beamSampleQuality="3" />
        <leg>
            <platformSettings x="{positions['leg1']['x']}" y="{positions['leg1']['y']}" z="{positions['leg1']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="90" headRotateStop_deg="270" trajectoryTimeInterval_s="3.0"/>
        </leg>
        <leg>
            <platformSettings x="{positions['leg2']['x']}" y="{positions['leg2']['y']}" z="{positions['leg2']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-40.0" verticalAngleMax_deg="60" headRotateStart_deg="-90" headRotateStop_deg="90" trajectoryTimeInterval_s="3.0"/>
        </leg>
		<leg>
            <platformSettings x="{positions['leg3']['x']}" y="{positions['leg3']['y']}" z="{positions['leg3']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-60.0" verticalAngleMax_deg="60" headRotateStart_deg="0" headRotateStop_deg="180" trajectoryTimeInterval_s="3.0"/>
        </leg>
		<leg>
            <platformSettings x="{positions['leg4']['x']}" y="{positions['leg4']['y']}" z="{positions['leg4']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-60.0" verticalAngleMax_deg="60" headRotateStart_deg="180" headRotateStop_deg="360" trajectoryTimeInterval_s="3.0"/>
        </leg>
		<leg>
            <platformSettings x="{positions['leg5']['x']}" y="{positions['leg5']['y']}" z="{positions['leg5']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-20.0" verticalAngleMax_deg="120" headRotateStart_deg="90" headRotateStop_deg="270" trajectoryTimeInterval_s="3.0"/>
        </leg>
		<leg>
            <platformSettings x="{positions['leg6']['x']}" y="{positions['leg6']['y']}" z="{positions['leg6']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-20.0" verticalAngleMax_deg="120" headRotateStart_deg="-90" headRotateStop_deg="90" trajectoryTimeInterval_s="3.0"/>
        </leg>
		<leg>
            <platformSettings x="{positions['leg7']['x']}" y="{positions['leg7']['y']}" z="{positions['leg7']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-100" verticalAngleMax_deg="0" headRotateStart_deg="0" headRotateStop_deg="180" trajectoryTimeInterval_s="3.0"/>
        </leg>
		<leg>
            <platformSettings x="{positions['leg8']['x']}" y="{positions['leg8']['y']}" z="{positions['leg8']['z']}" onGround="false" />
            <scannerSettings template="profile1" verticalAngleMin_deg="-100" verticalAngleMax_deg="0" headRotateStart_deg="180" headRotateStop_deg="360" trajectoryTimeInterval_s="3.0"/>
        </leg>
    </survey>
</document>"""
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    print(f"Created survey file: {output_path}")
