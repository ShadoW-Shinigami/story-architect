"""
Export Utilities
Functions for exporting pipeline outputs to various formats.
"""

from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import html
from loguru import logger


def generate_notion_markdown(
    shot_grouping_data: Dict[str, Any],
    shot_breakdown_data: Dict[str, Any]
) -> str:
    """
    Convert Agent 4 shot grouping JSON to Notion-compatible Markdown.
    Uses HTML <details> tags for collapsible parent/child hierarchy.

    Args:
        shot_grouping_data: Dictionary containing ShotGrouping JSON output from Agent 4
        shot_breakdown_data: Dictionary containing ShotBreakdown JSON output from Agent 3

    Returns:
        Formatted Markdown string with ALL data preserved (no truncation)
    """
    lines = []

    # Create shot lookup dictionary from Agent 3 output
    shots_by_id = {}
    for shot in shot_breakdown_data.get('shots', []):
        shot_id = shot.get('shot_id')
        if shot_id:
            shots_by_id[shot_id] = shot

    # Header
    lines.append("# Shot Grouping Export\n")

    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append(f"**Total Parent Shots:** {shot_grouping_data.get('total_parent_shots', 0)}\n")
    lines.append(f"**Total Child Shots:** {shot_grouping_data.get('total_child_shots', 0)}\n")

    strategy = shot_grouping_data.get('grouping_strategy', 'N/A')
    lines.append(f"\n**Grouping Strategy:** {strategy}\n")

    lines.append("\n---\n")

    # Process parent shots grouped by scene
    parent_shots = shot_grouping_data.get('parent_shots', [])

    # Group by scene_id for better organization
    shots_by_scene = {}
    for shot in parent_shots:
        scene_id = shot.get('scene_id', 'UNKNOWN')
        if scene_id not in shots_by_scene:
            shots_by_scene[scene_id] = []
        shots_by_scene[scene_id].append(shot)

    # Render each scene
    for scene_id, shots in sorted(shots_by_scene.items()):
        lines.append(f"\n## Scene: {scene_id}\n")

        for shot in shots:
            lines.append(_format_grouped_shot(shot, shots_by_id, level=0))
            lines.append("\n---\n")

    return "\n".join(lines)


def _format_grouped_shot(
    shot: Dict[str, Any],
    shots_by_id: Dict[str, Dict[str, Any]],
    level: int = 0
) -> str:
    """
    Recursively format a grouped shot with its children.
    Uses HTML <details> tags for collapsible sections.

    Args:
        shot: GroupedShot dictionary (from Agent 4)
        shots_by_id: Dictionary mapping shot_id to full Shot data (from Agent 3)
        level: Nesting level (for indentation)

    Returns:
        Markdown formatted string with complete shot data
    """
    indent = "  " * level  # 2 spaces per level
    lines = []

    # Get shot grouping details (from Agent 4)
    shot_id = shot.get('shot_id', 'UNKNOWN')
    shot_type = shot.get('shot_type', 'unknown')
    grouping_reason = shot.get('grouping_reason', 'N/A')
    child_shots = shot.get('child_shots', [])

    # Look up full shot data from Agent 3 output
    shot_data = shots_by_id.get(shot_id, {})
    shot_description = shot_data.get('shot_description', 'No description')

    # Start details block
    # Summary shows shot ID and full description (no truncation)
    summary = f"{shot_id} - {shot_description}"
    lines.append(f"{indent}<details>")
    lines.append(f"{indent}<summary><strong>{_escape_html(summary)}</strong></summary>")
    lines.append("")

    # Shot metadata
    lines.append(f"{indent}### Shot Details\n")
    lines.append(f"{indent}- **Shot ID:** {shot_id}")
    lines.append(f"{indent}- **Shot Type:** {shot_type.capitalize()}")

    parent_shot_id = shot.get('parent_shot_id')
    if parent_shot_id:
        lines.append(f"{indent}- **Parent Shot:** {parent_shot_id}")

    scene_id = shot.get('scene_id', 'UNKNOWN')
    lines.append(f"{indent}- **Scene:** {scene_id}")

    # Shot data fields
    location = shot_data.get('location', 'N/A')
    lines.append(f"{indent}- **Location:** {location}")

    characters = shot_data.get('characters', [])
    if characters:
        char_list = ", ".join(characters)
        lines.append(f"{indent}- **Characters:** {char_list}")
    else:
        lines.append(f"{indent}- **Characters:** None")

    dialogue = shot_data.get('dialogue')
    if dialogue:
        lines.append(f"{indent}- **Dialogue:** \"{_escape_html(dialogue)}\"")
    else:
        lines.append(f"{indent}- **Dialogue:** None")

    # First Frame (complete, no truncation)
    first_frame = shot_data.get('first_frame', 'N/A')
    lines.append(f"\n{indent}#### First Frame Description\n")
    lines.append(f"{indent}{first_frame}\n")

    # Animation (complete, no truncation)
    animation = shot_data.get('animation', 'N/A')
    lines.append(f"{indent}#### Animation Instructions\n")
    lines.append(f"{indent}{animation}\n")

    # Grouping Reason
    lines.append(f"{indent}#### Grouping Reason\n")
    lines.append(f"{indent}{grouping_reason}\n")

    # Child Shots (recursive)
    if child_shots:
        lines.append(f"{indent}### Child Shots\n")

        for child in child_shots:
            # Recursively format children (increase indent level, pass shot lookup)
            child_markdown = _format_grouped_shot(child, shots_by_id, level=level + 1)
            lines.append(child_markdown)
            lines.append("")

    # Close details block
    lines.append(f"{indent}</details>")

    return "\n".join(lines)


def _escape_html(text: str) -> str:
    """
    Escape HTML special characters to prevent breaking HTML tags.

    Args:
        text: Raw text string

    Returns:
        HTML-escaped text
    """
    return html.escape(str(text))


def generate_scene_breakdown_markdown(scene_breakdown_data: Dict[str, Any]) -> str:
    """
    Convert Agent 2 scene breakdown JSON to readable Markdown.

    Args:
        scene_breakdown_data: Dictionary containing SceneBreakdown JSON output

    Returns:
        Formatted Markdown string
    """
    lines = []

    lines.append("# Scene Breakdown Export\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append(f"**Total Scenes:** {scene_breakdown_data.get('total_scenes', 0)}\n")

    lines.append("\n---\n")

    # Process each scene
    scenes = scene_breakdown_data.get('scenes', [])

    for scene in scenes:
        scene_id = scene.get('scene_id', 'UNKNOWN')
        lines.append(f"\n## {scene_id}\n")

        # Location
        location = scene.get('location', {})
        lines.append(f"**Location:** {location.get('name', 'N/A')}\n")
        lines.append(f"{location.get('description', 'N/A')}\n")

        # Time of day
        time_of_day = scene.get('time_of_day')
        if time_of_day:
            lines.append(f"\n**Time of Day:** {time_of_day}\n")

        # Characters
        characters = scene.get('characters', [])
        if characters:
            lines.append(f"\n### Characters\n")
            for char in characters:
                char_name = char.get('name', 'Unknown')
                char_desc = char.get('description', 'N/A')
                lines.append(f"\n**{char_name}**\n{char_desc}\n")

        # Screenplay Text
        screenplay_text = scene.get('screenplay_text', 'N/A')
        lines.append(f"\n### Screenplay\n")
        lines.append(f"```\n{screenplay_text}\n```\n")

        # Subscenes
        subscenes = scene.get('subscenes', [])
        if subscenes:
            lines.append(f"\n### Subscenes\n")
            for subscene in subscenes:
                subscene_id = subscene.get('subscene_id', 'UNKNOWN')
                event = subscene.get('event', 'UNKNOWN')
                lines.append(f"\n**{subscene_id}** - {event}\n")

                if event == "CHARACTER_ADDED":
                    char_added = subscene.get('character_added', {})
                    if char_added:
                        lines.append(f"- **Character:** {char_added.get('name', 'Unknown')}\n")
                        lines.append(f"- **Description:** {char_added.get('description', 'N/A')}\n")

                excerpt = subscene.get('screenplay_excerpt', '')
                if excerpt:
                    lines.append(f"```\n{excerpt}\n```\n")

        lines.append("\n---\n")

    return "\n".join(lines)


def generate_shot_breakdown_markdown(shot_breakdown_data: Dict[str, Any]) -> str:
    """
    Convert Agent 3 shot breakdown JSON to readable Markdown.

    Args:
        shot_breakdown_data: Dictionary containing ShotBreakdown JSON output

    Returns:
        Formatted Markdown string
    """
    lines = []

    lines.append("# Shot Breakdown Export\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append(f"**Total Shots:** {shot_breakdown_data.get('total_shots', 0)}\n")

    lines.append("\n---\n")

    # Group shots by scene
    shots = shot_breakdown_data.get('shots', [])
    shots_by_scene = {}

    for shot in shots:
        scene_id = shot.get('scene_id', 'UNKNOWN')
        if scene_id not in shots_by_scene:
            shots_by_scene[scene_id] = []
        shots_by_scene[scene_id].append(shot)

    # Render each scene's shots
    for scene_id, scene_shots in sorted(shots_by_scene.items()):
        lines.append(f"\n## Scene: {scene_id}\n")

        for shot in scene_shots:
            shot_id = shot.get('shot_id', 'UNKNOWN')
            shot_desc = shot.get('shot_description', 'No description')

            lines.append(f"\n### {shot_id}\n")
            lines.append(f"{shot_desc}\n")

            # Location and Characters
            location = shot.get('location', 'N/A')
            characters = shot.get('characters', [])
            char_list = ", ".join(characters) if characters else "None"

            lines.append(f"- **Location:** {location}")
            lines.append(f"- **Characters:** {char_list}")

            # Dialogue
            dialogue = shot.get('dialogue')
            if dialogue:
                lines.append(f"- **Dialogue:** \"{dialogue}\"")
            else:
                lines.append(f"- **Dialogue:** None")

            # First Frame
            first_frame = shot.get('first_frame', 'N/A')
            lines.append(f"\n**First Frame:**\n{first_frame}\n")

            # Animation
            animation = shot.get('animation', 'N/A')
            lines.append(f"\n**Animation:**\n{animation}\n")

            lines.append("\n---\n")

    return "\n".join(lines)


def generate_complete_export_zip(session_dir: Path, session_data: Dict[str, Any]) -> Path:
    """
    Generate complete ZIP export with all outputs + assets folder.
    Creates Notion-importable structure with images.

    Args:
        session_dir: Path to session directory
        session_data: Complete session data with all agent outputs

    Returns:
        Path to generated ZIP file
    """
    import zipfile
    import shutil
    import tempfile

    logger.info(f"Generating complete export ZIP for session: {session_dir.name}")

    # Create temporary directory for export staging
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)

        # Copy all agent JSON outputs
        for agent_file in session_dir.glob("agent_*_output.*"):
            shutil.copy(agent_file, export_dir / agent_file.name)

        # Copy session state
        session_file = session_dir / "session_state.json"
        if session_file.exists():
            shutil.copy(session_file, export_dir / "session_state.json")

        # Copy assets folder if it exists (Phase 2 images)
        assets_src = session_dir / "assets"
        if assets_src.exists() and assets_src.is_dir():
            assets_dest = export_dir / "assets"
            shutil.copytree(assets_src, assets_dest)
            logger.info(f"Copied assets folder with {sum(1 for f in assets_src.rglob('*') if f.is_file())} files")

        # Generate comprehensive Markdown file
        markdown_path = export_dir / "STORY_ARCHITECT_EXPORT.md"
        markdown_content = _generate_comprehensive_markdown(session_data, session_dir)
        markdown_path.write_text(markdown_content, encoding='utf-8')

        # Create ZIP file
        zip_filename = f"{session_dir.name}_complete_export.zip"
        zip_path = session_dir / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in export_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(export_dir)
                    zipf.write(file_path, arcname)

        logger.info(f"Created export ZIP: {zip_filename} ({zip_path.stat().st_size / 1024:.1f} KB)")

        return zip_path


def generate_notion_export_zip(session_dir: Path, session_data: Dict[str, Any]) -> Path:
    """
    Generate Notion-compatible export ZIP (markdown + assets only, no JSON files).
    Creates clean structure for direct import into Notion.

    Args:
        session_dir: Path to session directory
        session_data: Complete session data with all agent outputs

    Returns:
        Path to generated ZIP file
    """
    import zipfile
    import shutil
    import tempfile

    logger.info(f"Generating Notion export ZIP for session: {session_dir.name}")

    # Create temporary directory for export staging
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir)

        # Copy ONLY assets folder (no JSON files)
        assets_src = session_dir / "assets"
        if assets_src.exists() and assets_src.is_dir():
            assets_dest = export_dir / "assets"
            shutil.copytree(assets_src, assets_dest)
            logger.info(f"Copied assets folder with {sum(1 for f in assets_src.rglob('*') if f.is_file())} files")

        # Generate comprehensive Markdown file with clean name
        markdown_path = export_dir / "Story_Architect_Export.md"
        markdown_content = _generate_comprehensive_markdown(session_data, session_dir)
        markdown_path.write_text(markdown_content, encoding='utf-8')

        # Create ZIP file
        zip_filename = f"{session_dir.name}_notion.zip"
        zip_path = session_dir / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in export_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(export_dir)
                    zipf.write(file_path, arcname)

        logger.info(f"Created Notion export ZIP: {zip_filename} ({zip_path.stat().st_size / 1024:.1f} KB)")

        return zip_path


def _image_to_base64(image_path: Path) -> str:
    """
    Convert image file to base64 data URI for HTML embedding.

    Args:
        image_path: Path to image file

    Returns:
        Base64 data URI string (data:image/png;base64,...)
    """
    import base64

    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        b64_data = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{b64_data}"
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {str(e)}")
        return ""


def generate_html_export(session_dir: Path, session_data: Dict[str, Any]) -> Path:
    """
    Generate single HTML file with all images embedded as base64 data URIs.
    Self-contained file that can be opened in any browser or imported to Notion.

    Args:
        session_dir: Path to session directory
        session_data: Complete session data with all agent outputs

    Returns:
        Path to generated HTML file
    """
    logger.info(f"Generating HTML export for session: {session_dir.name}")

    agents_data = session_data.get('agents', {})
    session_id = session_data.get('session_id', 'Unknown')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Build HTML content
    html_parts = []

    # HTML header with CSS
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Story Architect Export - {session_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a1a;
            margin-bottom: 10px;
            font-size: 2.5em;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 2em;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        .character, .shot {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }}
        .shot {{
            border-left-color: #2196F3;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .description {{
            margin-top: 10px;
            color: #555;
            font-style: italic;
        }}
        .status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .status.verified {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status.soft_failure {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Story Architect - Complete Export</h1>
        <div class="metadata">
            <strong>Session:</strong> {session_id}<br>
            <strong>Exported:</strong> {timestamp}
        </div>
""")

    # Characters section
    if "agent_5" in agents_data:
        char_out = agents_data["agent_5"].get("output_data", {})
        html_parts.append("\n        <h2>Characters</h2>\n")

        for char in char_out.get("characters", []):
            char_name = html.escape(char.get('name', 'Unknown'))
            char_desc = html.escape(char.get('description', ''))

            html_parts.append(f'        <div class="character">\n')
            html_parts.append(f'            <h3>{char_name}</h3>\n')

            # Embed character image
            if char.get("image_path"):
                img_path = session_dir / char["image_path"]
                if img_path.exists():
                    img_data = _image_to_base64(img_path)
                    if img_data:
                        html_parts.append(f'            <img src="{img_data}" alt="{char_name}">\n')

            if char_desc:
                html_parts.append(f'            <div class="description">{char_desc}</div>\n')

            html_parts.append('        </div>\n')

    # Generated Shots section - Sequential ordering
    if "agent_7" in agents_data:
        html_parts.append("\n        <h2>Generated Shots</h2>\n")

        # Combine parent and child shots for sequential display
        all_shots = []

        # Add parent shots
        parent_data = agents_data.get("agent_7", {}).get("output_data", {})
        for shot in parent_data.get("parent_shots", []):
            shot_copy = dict(shot)
            shot_copy['_display_type'] = 'Parent'
            all_shots.append(shot_copy)

        # Add child shots
        if "agent_9" in agents_data:
            child_data = agents_data.get("agent_9", {}).get("output_data", {})
            for shot in child_data.get("child_shots", []):
                shot_copy = dict(shot)
                shot_copy['_display_type'] = 'Child'
                all_shots.append(shot_copy)

        # Sort shots sequentially
        all_shots.sort(key=lambda s: _natural_sort_key(s.get('shot_id', '')))

        # Display in sequential order
        for shot in all_shots:
            shot_id = html.escape(shot.get('shot_id', 'Unknown'))
            shot_type = shot.get('_display_type', 'Unknown')
            status = shot.get('verification_status', 'unknown')

            status_class = 'verified' if status == 'verified' else 'soft_failure'

            html_parts.append(f'        <div class="shot">\n')
            html_parts.append(f'            <h3>{shot_id} ({shot_type})<span class="status {status_class}">{status}</span></h3>\n')

            # Embed shot image
            if shot.get("image_path"):
                img_path = session_dir / shot["image_path"]
                if img_path.exists():
                    img_data = _image_to_base64(img_path)
                    if img_data:
                        html_parts.append(f'            <img src="{img_data}" alt="{shot_id}">\n')

            html_parts.append('        </div>\n')

    # Footer
    html_parts.append("""
        <div class="footer">
            Generated by Story Architect
        </div>
    </div>
</body>
</html>
""")

    # Write HTML file
    html_filename = f"{session_dir.name}_export.html"
    html_path = session_dir / html_filename

    html_content = ''.join(html_parts)
    html_path.write_text(html_content, encoding='utf-8')

    file_size_kb = html_path.stat().st_size / 1024
    logger.info(f"Created HTML export: {html_filename} ({file_size_kb:.1f} KB)")

    return html_path


def generate_html_export_parts(session_dir: Path, session_data: Dict[str, Any]) -> List[Path]:
    """
    Generate HTML export auto-split into multiple parts (each < 50MB).
    Creates multiple self-contained HTML files if needed.

    Args:
        session_dir: Path to session directory
        session_data: Complete session data with all agent outputs

    Returns:
        List of Path objects for each generated HTML file
    """
    logger.info(f"Generating HTML export with auto-splitting for session: {session_dir.name}")

    MAX_SIZE_BYTES = 45 * 1024 * 1024  # 45MB threshold (safety margin for 50MB)

    agents_data = session_data.get('agents', {})
    session_id = session_data.get('session_id', 'Unknown')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Build CSS header (reused in all parts)
    css_header = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Story Architect Export - {session_id}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1a1a1a;
            margin-bottom: 10px;
            font-size: 2.5em;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #2c3e50;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 2em;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }
        h3 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .metadata {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 30px;
        }
        .character, .shot {
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }
        .shot {
            border-left-color: #2196F3;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .description {
            margin-top: 10px;
            color: #555;
            font-style: italic;
        }
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }
        .status.verified {
            background-color: #d4edda;
            color: #155724;
        }
        .status.soft_failure {
            background-color: #fff3cd;
            color: #856404;
        }
        .footer {
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
"""

    footer_html = """
        <div class="footer">
            Generated by Story Architect
        </div>
    </div>
</body>
</html>
"""

    # Build characters HTML (always in Part 1)
    characters_html = []
    if "agent_5" in agents_data:
        char_out = agents_data["agent_5"].get("output_data", {})
        characters_html.append("\n        <h2>Characters</h2>\n")

        for char in char_out.get("characters", []):
            char_name = html.escape(char.get('name', 'Unknown'))
            char_desc = html.escape(char.get('description', ''))

            characters_html.append(f'        <div class="character">\n')
            characters_html.append(f'            <h3>{char_name}</h3>\n')

            # Embed character image
            if char.get("image_path"):
                img_path = session_dir / char["image_path"]
                if img_path.exists():
                    img_data = _image_to_base64(img_path)
                    if img_data:
                        characters_html.append(f'            <img src="{img_data}" alt="{char_name}">\n')

            if char_desc:
                characters_html.append(f'            <div class="description">{char_desc}</div>\n')

            characters_html.append('        </div>\n')

    # Prepare all shots
    all_shots = []
    if "agent_7" in agents_data:
        parent_data = agents_data.get("agent_7", {}).get("output_data", {})
        for shot in parent_data.get("parent_shots", []):
            shot_copy = dict(shot)
            shot_copy['_display_type'] = 'Parent'
            all_shots.append(shot_copy)

        if "agent_9" in agents_data:
            child_data = agents_data.get("agent_9", {}).get("output_data", {})
            for shot in child_data.get("child_shots", []):
                shot_copy = dict(shot)
                shot_copy['_display_type'] = 'Child'
                all_shots.append(shot_copy)

        # Sort shots sequentially
        all_shots.sort(key=lambda s: _natural_sort_key(s.get('shot_id', '')))

    # Build parts list with size tracking
    parts = []
    current_part_content = []
    current_part_size = 0

    # Add characters to Part 1
    characters_content = ''.join(characters_html)
    characters_size = len(characters_content.encode('utf-8'))
    current_part_content.append(characters_content)
    current_part_size += characters_size

    # Add shots one by one, splitting when needed
    if all_shots:
        current_part_content.append("\n        <h2>Generated Shots</h2>\n")
        current_part_size += len("\n        <h2>Generated Shots</h2>\n".encode('utf-8'))

        for shot in all_shots:
            # Build shot HTML
            shot_id = html.escape(shot.get('shot_id', 'Unknown'))
            shot_type = shot.get('_display_type', 'Unknown')
            status = shot.get('verification_status', 'unknown')
            status_class = 'verified' if status == 'verified' else 'soft_failure'

            shot_html_parts = []
            shot_html_parts.append(f'        <div class="shot">\n')
            shot_html_parts.append(f'            <h3>{shot_id} ({shot_type})<span class="status {status_class}">{status}</span></h3>\n')

            # Embed shot image
            if shot.get("image_path"):
                img_path = session_dir / shot["image_path"]
                if img_path.exists():
                    img_data = _image_to_base64(img_path)
                    if img_data:
                        shot_html_parts.append(f'            <img src="{img_data}" alt="{shot_id}">\n')

            shot_html_parts.append('        </div>\n')

            shot_html = ''.join(shot_html_parts)
            shot_size = len(shot_html.encode('utf-8'))

            # Check if adding this shot would exceed limit
            # Also account for header/footer overhead (~10KB)
            overhead = len(css_header.encode('utf-8')) + len(footer_html.encode('utf-8'))

            if current_part_size + shot_size + overhead > MAX_SIZE_BYTES and len(parts) < 9:  # Max 10 parts
                # Save current part
                parts.append({
                    'content': current_part_content[:],
                    'size': current_part_size
                })

                # Start new part (shots only, no characters)
                current_part_content = ["\n        <h2>Generated Shots (continued)</h2>\n"]
                current_part_size = len("\n        <h2>Generated Shots (continued)</h2>\n".encode('utf-8'))

            current_part_content.append(shot_html)
            current_part_size += shot_size

    # Save final part
    parts.append({
        'content': current_part_content,
        'size': current_part_size
    })

    # Write all parts to files
    file_paths = []
    total_parts = len(parts)

    for part_num, part_data in enumerate(parts, 1):
        # Build part header with part info
        if total_parts > 1:
            header_html = f"""        <h1>Story Architect - Export (Part {part_num} of {total_parts})</h1>
        <div class="metadata">
            <strong>Session:</strong> {session_id}<br>
            <strong>Exported:</strong> {timestamp}<br>
            <strong>Part:</strong> {part_num} of {total_parts}
        </div>
"""
        else:
            header_html = f"""        <h1>Story Architect - Complete Export</h1>
        <div class="metadata">
            <strong>Session:</strong> {session_id}<br>
            <strong>Exported:</strong> {timestamp}
        </div>
"""

        # Build complete HTML
        html_content = css_header + header_html + ''.join(part_data['content']) + footer_html

        # Write file
        if total_parts > 1:
            html_filename = f"{session_dir.name}_export_part{part_num}.html"
        else:
            html_filename = f"{session_dir.name}_export.html"

        html_path = session_dir / html_filename
        html_path.write_text(html_content, encoding='utf-8')

        file_size_mb = html_path.stat().st_size / (1024 * 1024)
        logger.info(f"Created HTML part {part_num}/{total_parts}: {html_filename} ({file_size_mb:.1f} MB)")

        file_paths.append(html_path)

    return file_paths


def _natural_sort_key(shot_id: str) -> List:
    """
    Generate sort key for natural sorting of shot IDs.
    Handles SHOT_1_1, SHOT_1_2, SHOT_1_10 correctly.

    Args:
        shot_id: Shot ID string (e.g., "SHOT_1_10")

    Returns:
        List of mixed strings and integers for sorting
    """
    import re
    parts = re.split(r'(\d+)', shot_id)
    return [int(p) if p.isdigit() else p for p in parts]


def _generate_comprehensive_markdown(session_data: Dict[str, Any], session_dir: Path) -> str:
    """Generate comprehensive Markdown with all outputs and images."""
    lines = []

    # Header
    lines.append("# Story Architect - Complete Export\n")
    lines.append(f"**Session:** {session_data.get('session_id', 'Unknown')}\n")
    lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n---\n")

    agents_data = session_data.get('agents', {})

    # Characters
    if "agent_5" in agents_data:
        char_out = agents_data["agent_5"].get("output_data", {})
        lines.append("\n## Characters\n")

        for char in char_out.get("characters", []):
            lines.append(f"\n### {char.get('name', 'Unknown')}\n")
            if char.get("image_path"):
                lines.append(f"![{char['name']}]({char['image_path']})\n")
            lines.append(f"{char.get('description', '')}\n")

    # Generated Images - Sequential shot ordering
    if "agent_7" in agents_data:
        lines.append("\n## Generated Shots\n")

        # Combine parent and child shots for sequential display
        all_shots = []

        # Add parent shots with type label
        parent_data = agents_data.get("agent_7", {}).get("output_data", {})
        for shot in parent_data.get("parent_shots", []):
            shot_copy = dict(shot)
            shot_copy['_display_type'] = 'Parent'
            all_shots.append(shot_copy)

        # Add child shots with type label
        if "agent_9" in agents_data:
            child_data = agents_data.get("agent_9", {}).get("output_data", {})
            for shot in child_data.get("child_shots", []):
                shot_copy = dict(shot)
                shot_copy['_display_type'] = 'Child'
                all_shots.append(shot_copy)

        # Sort shots sequentially by shot_id (natural sort)
        all_shots.sort(key=lambda s: _natural_sort_key(s.get('shot_id', '')))

        # Display in sequential order
        for shot in all_shots:
            shot_id = shot.get('shot_id', 'Unknown')
            shot_type = shot.get('_display_type', 'Unknown')
            status = shot.get('verification_status', 'unknown')

            lines.append(f"\n### {shot_id} ({shot_type}, {status})\n")
            if shot.get("image_path"):
                lines.append(f"![{shot_id}]({shot['image_path']})\n")

    lines.append("\n---\n*Generated by Story Architect*\n")

    return "\n".join(lines)
