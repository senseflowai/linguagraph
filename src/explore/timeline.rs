//! Timeline extraction: turn `Datetime`-typed properties into a
//! chronologically sorted event list.
//!
//! Events come from each node's [`dates`](super::dto::PropertyGroups::dates)
//! bucket (catalog-classified `Datetime` properties, string or
//! epoch-seconds values). As a fallback for graphs without an ontology,
//! string values in the `other` bucket that strictly parse as ISO-8601
//! dates are picked up too — numbers outside the dates bucket are never
//! treated as epochs (a vote count is not a date).

use serde_json::Value as JsonValue;

use crate::types::handlers::core::epoch_to_ymdhms;

use super::dto::{NodeView, Subgraph, TimelineEvent};

/// Extract every dated event from a subgraph, sorted chronologically
/// (unparseable dates last, ordered by their raw string).
pub(crate) fn subgraph_timeline(subgraph: &Subgraph) -> Vec<TimelineEvent> {
    let mut events: Vec<TimelineEvent> = subgraph.nodes.iter().flat_map(node_events).collect();
    sort_events(&mut events);
    events
}

/// Extract events from a plain node list (entity-table pages).
pub(crate) fn nodes_timeline(nodes: &[NodeView]) -> Vec<TimelineEvent> {
    let mut events: Vec<TimelineEvent> = nodes.iter().flat_map(node_events).collect();
    sort_events(&mut events);
    events
}

fn sort_events(events: &mut [TimelineEvent]) {
    events.sort_by(|a, b| match (a.epoch_seconds, b.epoch_seconds) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.date.cmp(&b.date),
    });
}

fn node_events(node: &NodeView) -> Vec<TimelineEvent> {
    let mut events = Vec::new();
    let event = |property: &str, date: String, epoch: Option<i64>| TimelineEvent {
        date,
        epoch_seconds: epoch,
        property: property.to_string(),
        entity_id: node.id.clone(),
        entity_name: node.name.clone(),
        entity_type: node.entity_type.clone(),
    };

    for (name, value) in &node.properties.dates {
        match value {
            JsonValue::String(s) => {
                let epoch = parse_iso_epoch(s);
                events.push(event(name, s.clone(), epoch));
            }
            JsonValue::Number(n) => {
                if let Some(secs) = n.as_i64() {
                    events.push(event(name, format_epoch(secs), Some(secs)));
                }
            }
            _ => {}
        }
    }
    // Fallback for uncatalogued graphs: strict ISO strings only.
    for (name, value) in &node.properties.other {
        if let JsonValue::String(s) = value {
            if let Some(epoch) = parse_iso_epoch(s) {
                events.push(event(name, s.clone(), Some(epoch)));
            }
        }
    }
    events
}

fn format_epoch(secs: i64) -> String {
    let (y, m, d, h, mi, s) = epoch_to_ymdhms(secs);
    if h == 0 && mi == 0 && s == 0 {
        format!("{y:04}-{m:02}-{d:02}")
    } else {
        format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}")
    }
}

/// Parse `YYYY-MM-DD[<T| >HH:MM[:SS]…]` into epoch seconds. Strict on
/// the date part (shape and range) so identifiers never masquerade as
/// dates; the optional time tail is best-effort (zone suffixes ignored).
pub(crate) fn parse_iso_epoch(raw: &str) -> Option<i64> {
    let s = raw.trim();
    if s.len() < 10 {
        return None;
    }
    let date = &s[..10];
    let b = date.as_bytes();
    let shape_ok = b[0..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit);
    if !shape_ok {
        return None;
    }
    let y: i64 = date[0..4].parse().ok()?;
    let m: u32 = date[5..7].parse().ok()?;
    let d: u32 = date[8..10].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }

    let rest = &s[10..];
    let (h, mi, sec) = if rest.is_empty() {
        (0, 0, 0)
    } else {
        let body = rest.strip_prefix('T').or_else(|| rest.strip_prefix(' '))?;
        parse_time_tail(body)?
    };
    Some(days_from_civil(y, m, d) * 86_400 + i64::from(h) * 3600 + i64::from(mi) * 60 + i64::from(sec))
}

fn parse_time_tail(body: &str) -> Option<(u32, u32, u32)> {
    let core = match body.find(['Z', '+', '-']) {
        Some(i) => &body[..i],
        None => body,
    };
    let core = core.split_once('.').map(|(a, _)| a).unwrap_or(core);
    let mut parts = core.split(':');
    let h: u32 = parts.next()?.parse().ok()?;
    let mi: u32 = parts.next()?.parse().ok()?;
    let sec: u32 = match parts.next() {
        Some(p) => p.parse().ok()?,
        None => 0,
    };
    (h < 24 && mi < 60 && sec <= 60).then_some((h, mi, sec))
}

/// Days since the Unix epoch for a civil date — Howard Hinnant's
/// `days_from_civil`, the inverse of [`epoch_to_ymdhms`].
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400; // [0, 399]
    let mp = i64::from((m + 9) % 12); // [0, 11]
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explore::dto::PropertyGroups;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn node(id: &str, dates: Vec<(&str, JsonValue)>, other: Vec<(&str, JsonValue)>) -> NodeView {
        NodeView {
            id: id.to_string(),
            name: id.to_string(),
            entity_type: "Movie".to_string(),
            labels: vec!["Movie".to_string()],
            properties: PropertyGroups {
                dates: dates
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect::<BTreeMap<_, _>>(),
                other: other
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect::<BTreeMap<_, _>>(),
                ..Default::default()
            },
            confidence: None,
            ephemeral_handle: false,
        }
    }

    #[test]
    fn parse_iso_epoch_round_trips_epoch_to_ymdhms() {
        for secs in [0_i64, 951_782_400, 1_234_567_890, -86_400] {
            let (y, m, d, h, mi, s) = epoch_to_ymdhms(secs);
            let rendered = format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}");
            assert_eq!(parse_iso_epoch(&rendered), Some(secs), "for {rendered}");
        }
    }

    #[test]
    fn parse_iso_epoch_rejects_non_dates() {
        assert_eq!(parse_iso_epoch("not a date"), None);
        assert_eq!(parse_iso_epoch("2024-13-01"), None);
        assert_eq!(parse_iso_epoch("2024-001-ab"), None);
        assert_eq!(parse_iso_epoch("m1"), None);
    }

    #[test]
    fn events_sort_chronologically_with_unparseable_last() {
        let nodes = vec![
            node(
                "b",
                vec![("released", json!("2003-11-05"))],
                vec![("note", json!("around March"))], // not a date → ignored
            ),
            node("a", vec![("released", json!("1999-03-31"))], vec![]),
            node("c", vec![("released", json!("someday"))], vec![]),
        ];
        let events = nodes_timeline(&nodes);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].entity_id, "a");
        assert_eq!(events[1].entity_id, "b");
        assert_eq!(events[2].entity_id, "c");
        assert_eq!(events[2].epoch_seconds, None, "kept, sorted last");
    }

    #[test]
    fn epoch_number_dates_render_as_iso() {
        let nodes = vec![node("a", vec![("ingested_at", json!(0))], vec![])];
        let events = nodes_timeline(&nodes);
        assert_eq!(events[0].date, "1970-01-01");
        assert_eq!(events[0].epoch_seconds, Some(0));
    }

    #[test]
    fn other_bucket_strings_need_strict_iso_shape() {
        let nodes = vec![node(
            "a",
            vec![],
            vec![
                ("founded", json!("2015-06-01")), // picked up
                ("code", json!("A-2015-06")),     // ignored
                ("votes", json!(4500)),           // numbers never treated as epochs here
            ],
        )];
        let events = nodes_timeline(&nodes);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].property, "founded");
    }
}
