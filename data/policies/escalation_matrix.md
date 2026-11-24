# Escalation matrix

## Payments Ops

- Route unresolved UPI debits after three business days and confirmed duplicate charges here.
- For a cluster of AutoPay timeouts, include the banks, time range, and failed transaction count.

## Other queues

- Send late or disputed returns to Fulfilment with the order ID and inspection status.
- Send missing coins and spent-coin refund requests to Subscription Support with the payment reference.
- Send short-notice or clinic-cancelled appointments to Clinic Support with the booking reference.

## Handoff

- Mark active duplicate debits or same-day appointment changes as high priority.
- Include the case ID, what was checked, and the time of the next promised update.
- Add the earlier ticket reference if the customer has already reported the issue.
- Keep the case assigned to one owner until the receiving queue accepts it.
- Get explicit approval before opening a ticket, sending an email, or issuing a refund.
