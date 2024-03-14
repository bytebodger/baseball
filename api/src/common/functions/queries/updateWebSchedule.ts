import { IdentifyField } from '../../enums/IdentifyField.js';
import { Table } from '../../enums/Table.js';
import { updateTableByPrimaryKey } from './updateTableByPrimaryKey.js';

interface Fields {
   has_been_played?: boolean,
   html?: string,
   time_checked: number,
   time_processed?: number | null,
   time_retrieved?: number,
   web_schedule_id: number,
}

export const updateWebSchedule = async (fields: Fields) => {
   return await updateTableByPrimaryKey(Table.webSchedule, IdentifyField.webSchedule, fields);
}