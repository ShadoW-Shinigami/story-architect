"""
Async pipeline orchestrator.

Coordinates the execution of all 11 agents in sequence.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy import select

from app.database import get_db_context
from app.models.session import Session, SessionStatus
from app.models.agent_output import AgentOutput, AgentStatus
from app.core.progress_tracker import ProgressTracker
from app.core.config import get_config


# Agent execution order
AGENT_ORDER = [
    "agent_1",   # Screenplay Generator
    "agent_2",   # Scene Breakdown
    "agent_3",   # Shot Breakdown
    "agent_4",   # Shot Grouping
    "agent_5",   # Character Creator
    "agent_6",   # Parent Image Generator
    "agent_7",   # Parent Verification
    "agent_8",   # Child Image Generator
    "agent_9",   # Child Verification
    "agent_10",  # Video Dialogue Generator
    "agent_11",  # Intelligent Video Editor
]


class AsyncPipeline:
    """
    Async pipeline orchestrator.

    Manages:
    - Agent execution order
    - Input/output passing between agents
    - Session state updates
    - Progress tracking
    """

    def __init__(self, session_id: str, progress_tracker: ProgressTracker):
        self.session_id = session_id
        self.progress = progress_tracker
        self.config = get_config()
        self._agents = {}  # Lazy-loaded agent instances

    async def run(self, start_agent: str = "agent_1"):
        """
        Run the pipeline from the specified starting agent.

        Args:
            start_agent: Agent to start from (default: agent_1)
        """
        await self.progress.pipeline_started()

        try:
            # Get starting index
            start_idx = AGENT_ORDER.index(start_agent)

            # Run agents in sequence
            for agent_name in AGENT_ORDER[start_idx:]:
                await self._run_agent(agent_name)

            await self.progress.pipeline_completed()

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            await self.progress.pipeline_failed(str(e))
            raise

    async def resume_from_agent(self, agent_name: str):
        """
        Resume pipeline from a specific agent.

        Useful for retrying failed agents or regenerating outputs.
        """
        if agent_name not in AGENT_ORDER:
            raise ValueError(f"Invalid agent name: {agent_name}")

        await self.run(start_agent=agent_name)

    async def _run_agent(self, agent_name: str):
        """Run a single agent with retry logic."""
        await self.progress.agent_started(agent_name)

        # Update session current agent
        async with get_db_context() as db:
            result = await db.execute(
                select(Session).where(Session.id == self.session_id)
            )
            session = result.scalars().first()
            session.current_agent = agent_name

        agent_config = self.config.get_agent_config(agent_name)
        max_retries = agent_config.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                # Get agent instance
                agent = await self._get_agent(agent_name)

                # Get input data for this agent
                input_data = await self._get_agent_input(agent_name)

                # Get progress callback for per-item updates
                progress_callback = self.progress.get_progress_callback(agent_name)

                # Execute agent
                start_time = datetime.utcnow()
                output = await agent.execute(input_data, progress_callback)
                duration = (datetime.utcnow() - start_time).total_seconds()

                # Save output
                await self._save_agent_output(agent_name, output)

                # Get output summary for progress event
                summary = self._get_output_summary(agent_name, output)

                await self.progress.agent_completed(
                    agent_name,
                    output_summary=summary,
                    duration=duration
                )

                return output

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Agent {agent_name} attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    await self.progress.agent_retry(agent_name, attempt + 1, str(e))
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    await self.progress.agent_failed(agent_name, str(e), attempt + 1)

                    # Save failed status
                    async with get_db_context() as db:
                        output = AgentOutput(
                            session_id=self.session_id,
                            agent_name=agent_name,
                            status=AgentStatus.FAILED,
                            error_message=str(e),
                            retry_count=attempt + 1
                        )
                        db.add(output)

                    raise RuntimeError(f"Agent {agent_name} failed after {max_retries} attempts: {last_error}")

    async def _get_agent(self, agent_name: str):
        """
        Get or create an agent instance.

        Agents are lazy-loaded and cached.
        """
        if agent_name in self._agents:
            return self._agents[agent_name]

        # Use agent factory to create real or placeholder agents
        from app.agents.factory import create_agent

        agent = await create_agent(agent_name, self.session_id)
        self._agents[agent_name] = agent
        return agent

    async def _get_agent_input(self, agent_name: str):
        """
        Get input data for an agent based on previous agent outputs.
        """
        async with get_db_context() as db:
            # Get session for original input
            result = await db.execute(
                select(Session).where(Session.id == self.session_id)
            )
            session = result.scalars().first()

            if agent_name == "agent_1":
                # Agent 1 takes raw input text (string, not dict)
                return session.input_data

            elif agent_name == "agent_2":
                # Agent 2 takes screenplay text from agent 1 (string)
                output_1 = await self._get_output(db, "agent_1")
                # Agent 1 returns a string directly
                return output_1 if isinstance(output_1, str) else output_1.get("text", "")

            elif agent_name == "agent_3":
                # Agent 3 takes scene breakdown dict from agent 2
                output_2 = await self._get_output(db, "agent_2")
                return output_2  # Returns the full scene breakdown dict

            elif agent_name == "agent_4":
                # Agent 4 takes shot breakdown dict from agent 3
                output_3 = await self._get_output(db, "agent_3")
                return output_3  # Returns the full shot breakdown dict

            # Build input for Phase 2+ agents based on dependencies
            inputs = {}

            if agent_name in ["agent_5", "agent_6", "agent_7", "agent_8", "agent_9"]:
                inputs["scene_breakdown"] = await self._get_output(db, "agent_2")
                inputs["shot_breakdown"] = await self._get_output(db, "agent_3")
                inputs["shot_grouping"] = await self._get_output(db, "agent_4")

                if agent_name in ["agent_6", "agent_7", "agent_8", "agent_9"]:
                    inputs["character_grids"] = await self._get_output(db, "agent_5")

                if agent_name in ["agent_7"]:
                    inputs["parent_shots"] = await self._get_output(db, "agent_6")

                if agent_name in ["agent_8", "agent_9"]:
                    inputs["parent_shots"] = await self._get_output(db, "agent_7")

                if agent_name == "agent_9":
                    inputs["child_shots"] = await self._get_output(db, "agent_8")

            # Phase 3 agents
            if agent_name in ["agent_10", "agent_11"]:
                inputs["scene_breakdown"] = await self._get_output(db, "agent_2")
                inputs["shot_breakdown"] = await self._get_output(db, "agent_3")
                inputs["shot_grouping"] = await self._get_output(db, "agent_4")
                inputs["character_grids"] = await self._get_output(db, "agent_5")
                inputs["parent_shots"] = await self._get_output(db, "agent_7")
                inputs["child_shots"] = await self._get_output(db, "agent_9")

                if agent_name == "agent_11":
                    inputs["videos"] = await self._get_output(db, "agent_10")

            return inputs

    async def _get_output(self, db, agent_name: str) -> dict:
        """Get output data from a completed agent."""
        result = await db.execute(
            select(AgentOutput).where(
                AgentOutput.session_id == self.session_id,
                AgentOutput.agent_name == agent_name
            )
        )
        output = result.scalars().first()

        if not output or not output.output_data:
            raise RuntimeError(f"Required output from {agent_name} not found")

        return output.output_data

    async def _save_agent_output(self, agent_name: str, output: dict):
        """Save agent output to database."""
        async with get_db_context() as db:
            # Check if output already exists
            result = await db.execute(
                select(AgentOutput).where(
                    AgentOutput.session_id == self.session_id,
                    AgentOutput.agent_name == agent_name
                )
            )
            existing = result.scalars().first()

            if existing:
                existing.output_data = output
                existing.status = AgentStatus.COMPLETED
                existing.completed_at = datetime.utcnow()
            else:
                agent_output = AgentOutput(
                    session_id=self.session_id,
                    agent_name=agent_name,
                    status=AgentStatus.COMPLETED,
                    output_data=output,
                    output_summary=self._get_output_summary(agent_name, output),
                    completed_at=datetime.utcnow()
                )
                db.add(agent_output)

    def _get_output_summary(self, agent_name: str, output) -> str:
        """Generate a brief summary of agent output for display."""
        if agent_name == "agent_1":
            # Agent 1 returns a string directly
            text = output if isinstance(output, str) else output.get("text", "")
            return f"{len(text)} characters"

        elif agent_name == "agent_2":
            scenes = output.get("scenes", [])
            return f"{len(scenes)} scenes"

        elif agent_name == "agent_3":
            shots = output.get("shots", [])
            return f"{len(shots)} shots"

        elif agent_name == "agent_4":
            parents = output.get("parent_shots", [])
            children = sum(len(p.get("child_shots", [])) for p in parents)
            return f"{len(parents)} parent shots, {children} child shots"

        elif agent_name == "agent_5":
            chars = output.get("characters", [])
            grids = output.get("character_grids", [])
            return f"{len(chars)} characters, {len(grids)} grids"

        elif agent_name in ["agent_6", "agent_8"]:
            shots = output.get("parent_shots", output.get("child_shots", []))
            return f"{len(shots)} images generated"

        elif agent_name in ["agent_7", "agent_9"]:
            shots = output.get("parent_shots", output.get("child_shots", []))
            verified = sum(1 for s in shots if s.get("verification_status") == "verified")
            return f"{verified}/{len(shots)} verified"

        elif agent_name == "agent_10":
            videos = output.get("videos", [])
            return f"{len(videos)} videos generated"

        elif agent_name == "agent_11":
            scenes = output.get("scene_videos", [])
            return f"Master video + {len(scenes)} scene videos"

        return "Completed"
