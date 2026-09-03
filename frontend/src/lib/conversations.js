import { supabase } from './supabase'

function fromRow(row) {
  return {
    id: row.id,
    title: row.title || 'New conversation',
    messages: Array.isArray(row.messages) ? row.messages : [],
  }
}

export async function loadSavedConversations(userId) {
  if (!supabase) return []

  const { data, error } = await supabase
    .from('conversations')
    .select('id, title, messages, updated_at')
    .eq('user_id', userId)
    .order('updated_at', { ascending: false })

  if (error) throw error
  return (data || []).map(fromRow)
}

export async function saveConversation(conversation, userId) {
  if (!supabase) return

  const { error } = await supabase.from('conversations').upsert({
    id: conversation.id,
    user_id: userId,
    title: conversation.title || 'New conversation',
    messages: conversation.messages,
    updated_at: new Date().toISOString(),
  })

  if (error) throw error
}

export async function deleteSavedConversation(conversationId, userId) {
  if (!supabase) return

  const { error } = await supabase
    .from('conversations')
    .delete()
    .eq('id', conversationId)
    .eq('user_id', userId)

  if (error) throw error
}
