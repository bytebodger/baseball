import { IdentifyField } from '../../enums/IdentifyField.js';
import { Table } from '../../enums/Table.js';
import { updateByPrimaryKey } from './updateByPrimaryKey.js';

interface Columns {
   has_been_played?: boolean,
   html?: string,
   time_processed?: number | null,
   time_retrieved?: number,
   web_scheduled_id: number,
}

export const updateWebSchedule = async (columns: Columns) => {
   return await updateByPrimaryKey(Table.webSchedule, IdentifyField.webSchedule, columns);
}