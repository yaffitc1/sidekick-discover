"""Task organization and prioritization for insights.

Categorizes insights into actionable tasks with priorities.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


def categorize_insight(insight: Dict[str, Any]) -> str:
    """Categorize an insight into a task category.
    
    Args:
        insight: Insight dictionary
    
    Returns:
        Category name: "data_quality", "relationships", "kpis", or "optimization"
    """
    title = insight.get("title", "").lower()
    rationale = insight.get("rationale", "").lower()
    
    # Data quality keywords
    quality_keywords = [
        "missing", "null", "duplicate", "invalid", "outlier", "validation",
        "empty", "incomplete", "format", "type", "consistency"
    ]
    
    # Relationship keywords
    relationship_keywords = [
        "correlation", "relationship", "foreign key", "join", "reference",
        "link", "connection"
    ]
    
    # KPI keywords
    kpi_keywords = [
        "kpi", "metric", "measure", "indicator", "trend", "ratio", "percentage"
    ]
    
    # Check title and rationale
    text = f"{title} {rationale}"
    
    if any(kw in text for kw in quality_keywords):
        return "data_quality"
    elif any(kw in text for kw in relationship_keywords):
        return "relationships"
    elif any(kw in text for kw in kpi_keywords):
        return "kpis"
    else:
        return "optimization"


def assign_priority(insight: Dict[str, Any]) -> str:
    """Assign priority level to an insight.
    
    Args:
        insight: Insight dictionary
    
    Returns:
        Priority: "critical", "high", "medium", or "low"
    """
    severity = insight.get("severity", "low")
    score = insight.get("score", 0)
    
    # Map severity to priority
    if severity == "critical" or score >= 90:
        return "critical"
    elif severity == "high" or score >= 70:
        return "high"
    elif severity == "medium" or score >= 50:
        return "medium"
    else:
        return "low"


def create_actionable_task(insight: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Create an actionable task from an insight.
    
    Args:
        insight: Insight dictionary
        category: Category name
    
    Returns:
        Task dictionary with actionable information
    """
    task = {
        "id": insight.get("id", f"task_{hash(str(insight))}"),
        "title": insight.get("title", "Untitled task"),
        "category": category,
        "priority": assign_priority(insight),
        "severity": insight.get("severity", "low"),
        "score": insight.get("score", 0),
        "affected_columns": insight.get("affectedColumns", []),
        "affected_tables": insight.get("affectedTables", []),
        "rationale": insight.get("rationale", ""),
        "actions": generate_actions(insight, category),
    }
    
    return task


def generate_actions(insight: Dict[str, Any], category: str) -> List[str]:
    """Generate actionable next steps for an insight.
    
    Args:
        insight: Insight dictionary
        category: Category name
    
    Returns:
        List of actionable steps
    """
    actions = []
    title = insight.get("title", "").lower()
    rationale = insight.get("rationale", "").lower()
    
    if category == "data_quality":
        if "missing" in title or "null" in title:
            actions.append("Investigate root cause of missing data")
            actions.append("Consider data cleaning or imputation strategies")
            actions.append("Review data collection process")
        
        if "duplicate" in title:
            actions.append("Identify and remove duplicate records")
            actions.append("Review data pipeline for duplicate creation")
            actions.append("Implement unique constraints if appropriate")
        
        if "outlier" in title or "range" in rationale:
            actions.append("Validate outlier values are legitimate")
            actions.append("Consider data validation rules")
            actions.append("Document expected value ranges")
        
        if "format" in title or "type" in title:
            actions.append("Standardize data format")
            actions.append("Convert to appropriate data type")
            actions.append("Update data ingestion pipeline")
    
    elif category == "relationships":
        if "foreign key" in title or "join" in title:
            actions.append("Verify relationship is intentional")
            actions.append("Consider adding foreign key constraints")
            actions.append("Document relationship in data dictionary")
        
        if "correlation" in title:
            actions.append("Investigate causal relationship")
            actions.append("Consider feature engineering opportunities")
            actions.append("Review for potential data quality issues")
    
    elif category == "kpis":
        actions.append("Review business relevance of suggested KPI")
        actions.append("Validate calculation formula")
        actions.append("Implement KPI tracking if approved")
        actions.append("Set up monitoring and alerting")
    
    elif category == "optimization":
        actions.append("Review optimization opportunity")
        actions.append("Estimate impact and effort")
        actions.append("Prioritize in data engineering roadmap")
    
    # Default actions if none generated
    if not actions:
        actions.append("Review and assess impact")
        actions.append("Plan remediation if needed")
    
    return actions


def organize_tasks(
    insights: List[Dict[str, Any]],
    relationships: Optional[List[Dict[str, Any]]] = None,
    validation_checks: Optional[List[Dict[str, Any]]] = None,
    kpi_suggestions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Organize insights into actionable tasks by category and priority.
    
    Args:
        insights: List of insight dictionaries
        relationships: Optional list of relationship dictionaries
        validation_checks: Optional list of validation check dictionaries
        kpi_suggestions: Optional list of KPI suggestion dictionaries
    
    Returns:
        Organized tasks dictionary with categories and priorities
    """
    tasks = []
    
    # Convert insights to tasks
    for insight in insights:
        category = categorize_insight(insight)
        task = create_actionable_task(insight, category)
        tasks.append(task)
    
    # Convert validation checks to tasks
    if validation_checks:
        for check in validation_checks:
            # Create insight-like dict from validation check
            insight = {
                "id": f"validation_{check.get('column', 'unknown')}_{check.get('type', 'unknown')}",
                "title": check.get("message", "Validation issue"),
                "severity": check.get("severity", "low"),
                "score": check.get("severity", "low") == "high" and 80 or (
                    check.get("severity", "low") == "medium" and 60 or 40
                ),
                "rationale": check.get("message", ""),
                "affectedColumns": [check["column"]] if "column" in check else [],
            }
            category = categorize_insight(insight)
            task = create_actionable_task(insight, category)
            tasks.append(task)
    
    # Convert relationship suggestions to tasks
    if relationships:
        for rel in relationships:
            if rel.get("match_rate", 0) >= 0.7:
                insight = {
                    "id": f"relationship_{rel.get('source_table', '')}_{rel.get('target_table', '')}",
                    "title": f"Relationship detected: {rel.get('source_table', '')}.{rel.get('source_column', '')} -> {rel.get('target_table', '')}.{rel.get('target_column', '')}",
                    "severity": rel.get("confidence", "low") == "high" and "high" or "medium",
                    "score": int(rel.get("match_rate", 0) * 100),
                    "rationale": f"Match rate: {rel.get('match_rate', 0):.1%}, confidence: {rel.get('confidence', 'unknown')}",
                    "affectedColumns": [rel.get("source_column", ""), rel.get("target_column", "")],
                    "affectedTables": [rel.get("source_table", ""), rel.get("target_table", "")],
                }
                category = categorize_insight(insight)
                task = create_actionable_task(insight, category)
                tasks.append(task)
    
    # Convert KPI suggestions to tasks
    if kpi_suggestions:
        for kpi in kpi_suggestions[:10]:  # Limit to top 10
            insight = {
                "id": f"kpi_{kpi.get('kpi_type', 'unknown')}_{hash(str(kpi))}",
                "title": kpi.get("description", "Suggested KPI"),
                "severity": "low",
                "score": int(kpi.get("relevance_score", 0) * 100),
                "rationale": f"Formula: {kpi.get('formula', 'N/A')}",
                "affectedColumns": kpi.get("columns", []),
            }
            category = "kpis"
            task = create_actionable_task(insight, category)
            tasks.append(task)
    
    # Organize by category
    organized = {
        "data_quality": [],
        "relationships": [],
        "kpis": [],
        "optimization": [],
    }
    
    for task in tasks:
        category = task.get("category", "optimization")
        if category in organized:
            organized[category].append(task)
    
    # Sort each category by priority and score
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    
    for category in organized:
        organized[category].sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 3),
                -x.get("score", 0)
            )
        )
    
    # Summary
    summary = {
        "total_tasks": len(tasks),
        "by_category": {
            cat: len(tasks_list) for cat, tasks_list in organized.items()
        },
        "by_priority": {
            "critical": len([t for t in tasks if t.get("priority") == "critical"]),
            "high": len([t for t in tasks if t.get("priority") == "high"]),
            "medium": len([t for t in tasks if t.get("priority") == "medium"]),
            "low": len([t for t in tasks if t.get("priority") == "low"]),
        },
    }
    
    return {
        "tasks": organized,
        "summary": summary,
        "all_tasks": tasks,  # Flat list for easy iteration
    }





