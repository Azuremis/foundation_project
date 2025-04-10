#!/bin/bash

# Docker Compose management script
set -e

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Define services
SERVICES=("db" "web" "pgadmin")
COMPOSE_FILE="docker-compose.yml"

function show_usage {
    echo "Usage: $0 [command] [service]"
    echo ""
    echo "Commands:"
    echo "  start-all         Start all services"
    echo "  stop-all          Stop all services"
    echo "  restart-all       Restart all services"
    echo "  status            Show status of all services"
    echo "  start [service]   Start a specific service"
    echo "  stop [service]    Stop a specific service"
    echo "  restart [service] Restart a specific service"
    echo ""
    echo "Available services: ${SERVICES[*]}"
    exit 1
}

function is_valid_service {
    local service=$1
    for s in "${SERVICES[@]}"; do
        if [[ "$s" == "$service" ]]; then
            return 0
        fi
    done
    return 1
}

# Command handling
case "$1" in
    start-all)
        echo "Starting all services..."
        docker-compose -f $COMPOSE_FILE up -d
        ;;
    stop-all)
        echo "Stopping all services..."
        docker-compose -f $COMPOSE_FILE down
        ;;
    restart-all)
        echo "Restarting all services..."
        docker-compose -f $COMPOSE_FILE down
        docker-compose -f $COMPOSE_FILE up -d
        ;;
    status)
        echo "Services status:"
        docker-compose -f $COMPOSE_FILE ps
        ;;
    start)
        if [[ -z "$2" ]]; then
            echo "Error: Service name is required"
            show_usage
        fi
        
        if ! is_valid_service "$2"; then
            echo "Error: Invalid service '$2'"
            show_usage
        fi
        
        echo "Starting service: $2"
        docker-compose -f $COMPOSE_FILE up -d "$2"
        ;;
    stop)
        if [[ -z "$2" ]]; then
            echo "Error: Service name is required"
            show_usage
        fi
        
        if ! is_valid_service "$2"; then
            echo "Error: Invalid service '$2'"
            show_usage
        fi
        
        echo "Stopping service: $2"
        docker-compose -f $COMPOSE_FILE stop "$2"
        ;;
    restart)
        if [[ -z "$2" ]]; then
            echo "Error: Service name is required"
            show_usage
        fi
        
        if ! is_valid_service "$2"; then
            echo "Error: Invalid service '$2'"
            show_usage
        fi
        
        echo "Restarting service: $2"
        docker-compose -f $COMPOSE_FILE restart "$2"
        ;;
    *)
        show_usage
        ;;
esac

echo "Done!" 