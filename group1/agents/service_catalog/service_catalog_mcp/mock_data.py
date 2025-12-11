# Mock service data
services = [
    {
        "id": 1,
        "name": "Office 365",
        "category": "Productivity",
        "description": ("Complete office suite including Word, Excel, PowerPoint, "
                        " and Teams"),
        "access_requirements": [
            "Active employment status",
            "Manager approval",
            "IT security training completion"
        ],
        "required_data": [
            "Employee ID",
            "Department",
            "Job role",
            "Manager email"
        ],
        "provisioning_time": "1-2 business days",
        "cost_center": "IT Services"
    },
    {
        "id": 2,
        "name": "Salesforce CRM",
        "category": "Sales & Marketing",
        "description": "Customer relationship management platform for sales teams",
        "access_requirements": [
            "Sales department membership",
            "Salesforce training certification",
            "Data privacy acknowledgment",
            "Director approval"
        ],
        "required_data": [
            "Employee ID",
            "Sales territory",
            "Role level (SDR, AE, Manager)",
            "Direct manager email",
            "Cost center code"
        ],
        "provisioning_time": "3-5 business days",
        "cost_center": "Sales Operations"
    },
    {
        "id": 3,
        "name": "AWS Development Account",
        "category": "Development",
        "description": "Amazon Web Services development environment access",
        "access_requirements": [
            "Engineering department membership",
            "AWS security training",
            "Technical lead approval",
            "Project assignment"
        ],
        "required_data": [
            "Employee ID",
            "Technical competency level",
            "Project code",
            "Team lead email",
            "Required AWS services list"
        ],
        "provisioning_time": "2-3 business days",
        "cost_center": "Engineering"
    },
    {
        "id": 4,
        "name": "HR Information System",
        "category": "Human Resources",
        "description": "Employee data management and HR processes",
        "access_requirements": [
            "HR department or manager role",
            "GDPR compliance training",
            "HR director approval",
            "Background check clearance"
        ],
        "required_data": [
            "Employee ID",
            "HR role type",
            "Access level required (read/write/admin)",
            "HR director email",
            "Justification document"
        ],
        "provisioning_time": "5-7 business days",
        "cost_center": "Human Resources"
    },
    {
        "id": 5,
        "name": "Financial Reporting Dashboard",
        "category": "Finance",
        "description": "Real-time financial metrics and reporting tools",
        "access_requirements": [
            "Finance team membership or C-level role",
            "Financial data handling training",
            "CFO approval",
            "Multi-factor authentication setup"
        ],
        "required_data": [
            "Employee ID",
            "Financial access level",
            "Department budget codes",
            "Manager email",
            "Mobile phone number for MFA"
        ],
        "provisioning_time": "3-4 business days",
        "cost_center": "Finance"
    }
]
