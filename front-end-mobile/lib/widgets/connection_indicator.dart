import 'package:flutter/material.dart';

/// Connection status indicator widget.
/// 
/// Displays the current connection status to the FastAPI backend
/// with color-coded visual feedback.
class ConnectionIndicator extends StatelessWidget {
  final bool isConnected;
  final String backendUrl;

  const ConnectionIndicator({
    super.key,
    required this.isConnected,
    required this.backendUrl,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: isConnected 
            ? Colors.green.withOpacity(0.9) 
            : Colors.red.withOpacity(0.9),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            isConnected ? Icons.cloud_done : Icons.cloud_off,
            color: Colors.white,
            size: 16,
          ),
          const SizedBox(width: 6),
          Text(
            isConnected ? 'Connected' : 'Disconnected',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
